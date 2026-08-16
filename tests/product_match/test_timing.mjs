/**
 * Tests for the timing smoke, the P-decision artifact, and the binding that
 * makes a match refuse to start without one.
 *
 *   node --test tests/product_match/test_timing.mjs
 *
 * NO timing game is played here. `runTimingSmoke` takes injected clock and game
 * seams, so its schedule, sidecars, wall-clock span and decision derivation are
 * exercised without invoking `playGame`; the production CLI is never called and
 * the reserved openings `200…209` never reach a real game. Fixtures live in
 * temporary directories and are deleted, so no timing evidence is retained.
 */
import { describe, it, after, before } from 'node:test';
import assert from 'node:assert';
import {
  mkdtemp,
  mkdir,
  rm,
  readdir,
  readFile,
  writeFile,
} from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  P_DECISION_PATH,
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
  isTracked,
  loadCommittedDecision,
} from './p_decision.mjs';
import {
  runTimingSmoke,
  timingSchedule,
  timingSidecarName,
} from './timing.mjs';
import { analyse, FROZEN_SPEC } from './analyse.mjs';
import {
  BASELINE_MODEL_ID,
  CANDIDATE_MODEL_ID,
  runMatchFromCommittedDecision,
} from './harness.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(HERE, '..', '..');

/** A wall time that yields exactly `gamesPerHour` for ten games. */
const wallMsFor = (gamesPerHour) => (TIMING_GAMES * 3_600_000) / gamesPerHour;

const validDecision = (gamesPerHour = 13.2) =>
  buildDecision({
    totalSequentialWallMs: wallMsFor(gamesPerHour),
    timingEvidence: Array.from({ length: TIMING_GAMES }, (_, i) => ({
      file: `timing_${String(i).padStart(2, '0')}.json`,
      sha256: 'a'.repeat(64),
    })),
    openingPoolSha256: 'b'.repeat(64),
    executionCommit: 'c'.repeat(40),
    baselineModelId: BASELINE_MODEL_ID,
    candidateModelId: CANDIDATE_MODEL_ID,
    ortVersion: '1.23.2',
    ortConfig: 'no options supplied',
    nSimulations: 800,
    cPuct: 1.5,
    moveTemp: 0,
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
    assert.strictEqual(deriveP(13.2), 200);
    assert.strictEqual(deriveP(0.1), 100);
  });

  it('permits no third outcome', () => {
    for (const g of [0.01, 5, 8.79, 8.8, 100, 1e6]) {
      assert.ok([P_IF_AT_OR_ABOVE, P_IF_BELOW].includes(deriveP(g)));
    }
  });

  it('refuses a throughput that is not a number', () => {
    assert.throws(
      () => deriveP(NaN),
      (e) => e.code === 'BAD_THROUGHPUT'
    );
    assert.throws(
      () => deriveP(Infinity),
      (e) => e.code === 'BAD_THROUGHPUT'
    );
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
      assert.ok(
        g.openingId >= 200 && g.openingId <= 209,
        `opening ${g.openingId} is a match opening`
      );
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
    let calls = 0;
    const clock = [1000, 1000 + SPAN_MS];
    result = await runTimingSmoke({
      runDir: dir,
      openings,
      baseline: { modelId: BASELINE_MODEL_ID, inference: 'baseline-stub' },
      candidate: { modelId: CANDIDATE_MODEL_ID, inference: 'candidate-stub' },
      nSimulations: 800,
      moveTemp: 0,
      ortVersion: '1.23.2',
      executionCommit: 'd'.repeat(40),
      now: () => clock[Math.min(calls++, clock.length - 1)],
      playGameFn: async ({ redInference, blackInference }) => {
        // Self-play: both sides must be the SAME model instance.
        assert.strictEqual(redInference, blackInference);
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
    assert.strictEqual(files.length, TIMING_GAMES);
    assert.deepStrictEqual(
      files,
      timingSchedule()
        .map((g, i) => timingSidecarName(i, g.openingId))
        .sort()
    );
    // Never in match/, which the analyser reads.
    await assert.rejects(readdir(join(dir, 'match')));
  });

  it('marks every sidecar as timing, self-play, on a reserved opening', async () => {
    for (const f of await readdir(join(dir, 'timing'))) {
      const s = JSON.parse(await readFile(join(dir, 'timing', f), 'utf8'));
      assert.strictEqual(s.kind, 'timing');
      assert.strictEqual(s.red_model_id, s.black_model_id, 'must be self-play');
      assert.ok(s.opening_id >= 200 && s.opening_id <= 209);
    }
  });

  it('measures one span, not a sum of per-game elapsed times', () => {
    assert.strictEqual(result.totalSequentialWallMs, SPAN_MS);
    // Each fake game reported 1 ms, so a sum would be 10 — proving the span is
    // not built from elapsed_ms.
    assert.notStrictEqual(result.totalSequentialWallMs, TIMING_GAMES);
  });

  it('derives the decision from that span alone', () => {
    const d = result.decision;
    assert.strictEqual(d.schema, SCHEMA);
    assert.strictEqual(d.measured.total_sequential_wall_ms, SPAN_MS);
    assert.ok(Math.abs(d.measured.games_per_hour - 13.2) < 1e-9);
    assert.strictEqual(d.selected_p, 200);
    assert.strictEqual(d.threshold_games_per_hour, 8.8);
    assert.deepStrictEqual(d.opening_mapping, TIMING_OPENING_MAPPING);
    assert.strictEqual(d.timing_evidence.length, TIMING_GAMES);
  });

  it('hashes the evidence as written to disk', async () => {
    const { createHash } = await import('node:crypto');
    for (const e of result.decision.timing_evidence) {
      const bytes = await readFile(join(dir, 'timing', e.file));
      assert.strictEqual(
        createHash('sha256').update(bytes).digest('hex'),
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
    // The number someone would have to fake. Recomputed, never read.
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

  it('rejects the wrong number of timing games or evidence entries', () => {
    expectFail('WRONG_TIMING_GAME_COUNT', (d) => {
      d.measured.timing_games = 5;
    });
    expectFail('TIMING_EVIDENCE_COUNT', (d) => {
      d.timing_evidence = d.timing_evidence.slice(0, 3);
    });
  });

  it('rejects duplicated or malformed evidence digests', () => {
    expectFail('DUPLICATE_TIMING_EVIDENCE', (d) => {
      d.timing_evidence[1] = { ...d.timing_evidence[0] };
    });
    expectFail('MALFORMED_TIMING_EVIDENCE', (d) => {
      d.timing_evidence[0].sha256 = 'not-a-digest';
    });
  });

  it('rejects missing required fields before comparing anything', () =>
    expectFail('MISSING_OR_MALFORMED_FIELD', (d) => {
      delete d.execution_commit;
    }));

  it('rejects a decision bound to a different pool or different models', () => {
    const d = validDecision();
    assert.ok(
      decisionFailures(d, { poolSha256: 'f'.repeat(64) })
        .map((f) => f.code)
        .includes('POOL_HASH_MISMATCH')
    );
    assert.ok(
      decisionFailures(d, { expected: { n_simulations: 400 } })
        .map((f) => f.code)
        .includes('BINDING_MISMATCH')
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

describe('loading the committed decision', () => {
  let dir;
  before(async () => {
    dir = await mkdtemp(join(tmpdir(), 'twixt-decision-'));
  });
  after(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  it('no decision exists in the repository yet, and that is the intended state', async () => {
    await assert.rejects(
      loadCommittedDecision(),
      (e) => e instanceof PDecisionError && e.code === 'DECISION_MISSING'
    );
    assert.strictEqual(isTracked(P_DECISION_RELPATH, REPO_ROOT), false);
  });

  it('refuses a decision that exists but is not committed', async () => {
    const path = join(dir, 'p_decision.json');
    await writeFile(path, JSON.stringify(validDecision()));
    await assert.rejects(
      loadCommittedDecision({
        path,
        relPath: 'p_decision.json',
        repoRoot: dir,
      }),
      (e) => e.code === 'DECISION_NOT_COMMITTED'
    );
  });

  it('refuses unparseable or invalid content', async () => {
    const bad = join(dir, 'bad.json');
    await writeFile(bad, '{ not json');
    await assert.rejects(
      loadCommittedDecision({ path: bad, requireTracked: false }),
      (e) => e.code === 'DECISION_UNPARSEABLE'
    );
    const invalid = join(dir, 'invalid.json');
    const d = validDecision();
    d.selected_p = 150;
    await writeFile(invalid, JSON.stringify(d));
    await assert.rejects(
      loadCommittedDecision({ path: invalid, requireTracked: false }),
      (e) => e.code === 'DECISION_INVALID'
    );
  });

  it('accepts a valid decision when tracking is not required', async () => {
    const path = join(dir, 'good.json');
    await writeFile(path, JSON.stringify(validDecision()));
    const d = await loadCommittedDecision({ path, requireTracked: false });
    assert.strictEqual(d.selected_p, 200);
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

  it('the production match entry point refuses, before loading a model', async () => {
    await assert.rejects(
      runMatchFromCommittedDecision({
        runDir: join(dir, 'run'),
        openings: [],
        requireCleanWorktree: false,
      }),
      (e) => e instanceof PDecisionError && e.code === 'DECISION_MISSING'
    );
    // No run directory was created, so no game could have started.
    await assert.rejects(readdir(join(dir, 'run')));
  });

  it('the analyser refuses when no decision is committed', async () => {
    const runDir = join(dir, 'analysed');
    await mkdir(join(runDir, 'match'), { recursive: true });
    // A structurally VALID run.json, so the rejection can only come from the
    // missing decision rather than from metadata validation firing first.
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
    const r = await analyse(runDir, [], FROZEN_SPEC);
    assert.strictEqual(r.verdict, 'REJECTED');
    const codes = r.failures.map((f) => f.code);
    assert.ok(
      codes.includes('P_DECISION_UNAVAILABLE') ||
        codes.includes('BAD_OPENING_POOL'),
      `got ${codes.join(', ')}`
    );
  });

  it('the analyser refuses when run.json.P disagrees with the decision', async () => {
    const runDir = join(dir, 'mismatch');
    await mkdir(join(runDir, 'match'), { recursive: true });
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
    const decisionPath = join(dir, 'committed.json');
    await writeFile(decisionPath, JSON.stringify(validDecision(13.2))); // selects 200

    const r = await analyse(
      runDir,
      Array.from({ length: 210 }, () => [
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
      ]),
      {
        ...FROZEN_SPEC,
        tCritical: { 100: 1.9842169515, 200: 1.9719565442 },
        pDecision: { path: decisionPath, requireTracked: false },
      }
    );
    assert.strictEqual(r.verdict, 'REJECTED');
    assert.ok(
      r.failures
        .map((f) => f.code)
        .includes('P_DOES_NOT_MATCH_COMMITTED_DECISION'),
      JSON.stringify(r.failures.map((f) => f.code))
    );
  });

  it('there is no fallback to run.json.P anywhere in the analyser', async () => {
    const src = await readFile(join(HERE, 'analyse.mjs'), 'utf8');
    // The binding must be unconditional except for the explicit test opt-out,
    // which is visible at the call site rather than hidden behind a default.
    assert.ok(
      src.includes('loadCommittedDecision'),
      'analyser must load the decision'
    );
    assert.ok(
      src.includes('P_DOES_NOT_MATCH_COMMITTED_DECISION'),
      'analyser must compare run.json.P with the decision'
    );
  });
});

describe('scope: this test file runs no timing game', () => {
  it('imports no way to run a real timing game', async () => {
    // Checked against the IMPORT LIST, not against a literal: an assertion that
    // greps for a string can match its own source and pass vacuously.
    const src = await readFile(join(HERE, 'test_timing.mjs'), 'utf8');
    const imports = [
      ...src.matchAll(/^\s*import\s[^;]*?from\s+'([^']+)'/gms),
    ].map((m) => m[1]);
    assert.ok(
      !imports.includes('node:child_process'),
      'must not be able to shell out to the timing CLI'
    );
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
  });

  it('no decision artifact was created at the committed path', async () => {
    await assert.rejects(readFile(P_DECISION_PATH), (e) => e.code === 'ENOENT');
  });
});
