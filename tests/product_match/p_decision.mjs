#!/usr/bin/env node
/**
 * The committed P-decision artifact: schema, derivation, and validator.
 *
 * `P` is chosen from TIMING ALONE and must be fixed before the first match game.
 * Binding it to a committed artifact — rather than to `run.json`, which a match
 * writes about itself — is what stops the sample size being decided, or
 * revised, by anything the match observed.
 *
 * Deliberately standalone: the analyser must enforce this binding without
 * importing the harness, so nothing here loads a model or plays a game.
 *
 * The artifact does not exist yet and is not created by this file. No
 * placeholder is provided: a file that looked like a decision but was not one
 * would be worse than none.
 *
 * Specification: docs/superpowers/2026-08-14-product-stack-comparison-specification.md §7.3
 */
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
export const REPO_ROOT = join(HERE, '..', '..');

/** The one location a decision may live. Not configurable. */
export const P_DECISION_RELPATH = 'tests/product_match/p_decision.json';
export const P_DECISION_PATH = join(REPO_ROOT, P_DECISION_RELPATH);
export const POOL_RELPATH = 'tests/product_match/openings.json';

export const SCHEMA = 'twixt-p-decision/1';

/** Frozen by specification §7.3. */
export const THRESHOLD_GAMES_PER_HOUR = 8.8;
export const TIMING_GAMES = 10;
export const P_IF_AT_OR_ABOVE = 200;
export const P_IF_BELOW = 100;

/**
 * The exact opening mapping. Self-play on both sides is what makes the smoke
 * OUTCOME-BLIND: a model playing itself yields no comparative information, so
 * `P` cannot be chosen with knowledge of the matchup.
 */
export const TIMING_OPENING_MAPPING = Object.freeze({
  baseline_self_play: Object.freeze([200, 201, 202, 203, 204]),
  candidate_self_play: Object.freeze([205, 206, 207, 208, 209]),
});

export const timingSidecarName = (index, openingId) =>
  `timing_${String(index).padStart(2, '0')}_opening_${openingId}.json`;

/** The ten timing games, in fixed order. */
export function timingSchedule() {
  const schedule = [];
  for (const openingId of TIMING_OPENING_MAPPING.baseline_self_play) {
    schedule.push({ openingId, arm: 'baseline_self_play' });
  }
  for (const openingId of TIMING_OPENING_MAPPING.candidate_self_play) {
    schedule.push({ openingId, arm: 'candidate_self_play' });
  }
  return schedule;
}

/** Exactly which evidence filenames a valid decision must carry. */
export const expectedTimingFilenames = () =>
  timingSchedule().map((g, i) => timingSidecarName(i, g.openingId));

/**
 * Files whose bytes determine how a game is played or scored.
 *
 * Commit ancestry is not enough: every descendant of the timing commit passes
 * an ancestry check, including one that rewrote MCTS, inference, the readout
 * policy or the harness afterwards. Such a timing measurement no longer
 * describes the match code. Digesting the surface catches that directly.
 */
export const EXECUTION_SURFACE_FILES = Object.freeze([
  'server/gameLogic.js',
  'tests/product_match/analyse.mjs',
  'server/inference.js',
  'server/mcts.js',
  'server/model_manifest.js',
  'server/readout_policy.js',
  'tests/product_match/generate_openings.mjs',
  'tests/product_match/harness.mjs',
  'tests/product_match/p_decision.mjs',
  'tests/product_match/timing.mjs',
]);

export class PDecisionError extends Error {
  constructor(code, message) {
    super(message);
    this.name = 'PDecisionError';
    this.code = code;
  }
}

export const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');

/**
 * A digest over the execution-critical surface, as committed at `commit`.
 *
 * Read from git rather than the working tree so it describes what was
 * committed, and so it can be recomputed for a past commit during analysis.
 */
export function executionSurfaceDigest(commit = 'HEAD', repoRoot = REPO_ROOT) {
  const parts = [];
  for (const rel of EXECUTION_SURFACE_FILES) {
    parts.push(`${rel}:${sha256(readCommittedBlob(rel, commit, repoRoot))}`);
  }
  return sha256(parts.join('\n'));
}

/**
 * Refuse to proceed if any execution-critical file differs from HEAD.
 *
 * Validating a decision against committed bytes while EXECUTING modified
 * working-tree modules would make the surface binding decorative: the digest
 * would describe code that is not the code running.
 */
export function assertCleanExecutionSurface(repoRoot = REPO_ROOT) {
  let dirty;
  try {
    dirty = execFileSync(
      'git',
      ['status', '--porcelain', '--', ...EXECUTION_SURFACE_FILES],
      {
        cwd: repoRoot,
      }
    )
      .toString()
      .trim();
  } catch (err) {
    throw new PDecisionError('SURFACE_STATUS_UNAVAILABLE', err.message);
  }
  if (dirty !== '') {
    throw new PDecisionError(
      'EXECUTION_SURFACE_DIRTY',
      `execution-critical files differ from HEAD, so the running code is not the committed code:\n${dirty}`
    );
  }
}

/**
 * Read a path's bytes AS COMMITTED at `commit`.
 *
 * `git ls-files` is not enough: it succeeds for a newly staged file and for a
 * tracked file whose working-tree bytes have since been edited, and reading the
 * working tree would then read mutable content. Reading the blob is the only
 * form of "committed" that cannot be edited underneath the check.
 */
export function readCommittedBlob(
  relPath,
  commit = 'HEAD',
  repoRoot = REPO_ROOT
) {
  try {
    return execFileSync('git', ['show', `${commit}:${relPath}`], {
      cwd: repoRoot,
      maxBuffer: 64 * 1024 * 1024,
    });
  } catch {
    throw new PDecisionError(
      'NOT_COMMITTED',
      `${relPath} is not committed at ${commit}; a staged or working-tree-only file is not a ` +
        `commitment, because its bytes can change after this check`
    );
  }
}

/** §7.3 throughput: ONE wall-clock span, never a sum of per-game times. */
export function computeThroughput(
  totalSequentialWallMs,
  nGames = TIMING_GAMES
) {
  if (!Number.isFinite(totalSequentialWallMs) || totalSequentialWallMs <= 0) {
    throw new PDecisionError(
      'BAD_WALL_TIME',
      'total_sequential_wall_ms must be positive'
    );
  }
  return (nGames * 3_600_000) / totalSequentialWallMs;
}

/** The whole of the choice: `≥ 8.8 → 200`, `< 8.8 → 100`. */
export function deriveP(gamesPerHour) {
  if (!Number.isFinite(gamesPerHour)) {
    throw new PDecisionError(
      'BAD_THROUGHPUT',
      'games_per_hour is not a finite number'
    );
  }
  return gamesPerHour >= THRESHOLD_GAMES_PER_HOUR
    ? P_IF_AT_OR_ABOVE
    : P_IF_BELOW;
}

const TOP_LEVEL_TYPES = {
  schema: 'string',
  selected_p: 'integer',
  threshold_games_per_hour: 'number',
  derivation: 'string',
  opening_pool_sha256: 'string',
  execution_commit: 'string',
  execution_surface_sha256: 'string',
  baseline_model_id: 'string',
  candidate_model_id: 'string',
  ort_version: 'string',
  ort_config: 'string',
  n_simulations: 'integer',
  c_puct: 'number',
  move_temp: 'number',
  execution_mode: 'string',
};

const MEASURED_TYPES = {
  timing_games: 'integer',
  total_sequential_wall_ms: 'number',
  games_per_hour: 'number',
};

const typeOk = (v, t) => {
  if (v === undefined || v === null) return false;
  if (t === 'string') return typeof v === 'string' && v.length > 0;
  if (t === 'number') return typeof v === 'number' && Number.isFinite(v);
  if (t === 'integer') return Number.isInteger(v);
  return false;
};

/**
 * Validate a decision. Returns a list of failures; empty means valid.
 *
 * Everything derivable is RE-DERIVED. `selected_p` and `games_per_hour` are
 * exactly the numbers someone would have to fake, so both are recomputed from
 * the measured wall time and compared.
 */
export function decisionFailures(
  decision,
  { poolSha256, surfaceSha256, expected = {} } = {}
) {
  const out = [];
  const fail = (code, detail) => out.push({ code, detail });

  if (
    decision === null ||
    typeof decision !== 'object' ||
    Array.isArray(decision)
  ) {
    return [{ code: 'DECISION_NOT_AN_OBJECT', detail: {} }];
  }

  for (const [field, type] of Object.entries(TOP_LEVEL_TYPES)) {
    if (!typeOk(decision[field], type)) {
      fail('MISSING_OR_MALFORMED_FIELD', {
        field,
        expected: type,
        found: decision[field] ?? null,
      });
    }
  }
  const m = decision.measured;
  if (m === null || typeof m !== 'object' || Array.isArray(m)) {
    fail('MISSING_MEASURED_BLOCK', {});
  } else {
    for (const [field, type] of Object.entries(MEASURED_TYPES)) {
      if (!typeOk(m[field], type)) {
        fail('MISSING_OR_MALFORMED_FIELD', {
          field: `measured.${field}`,
          expected: type,
        });
      }
    }
  }
  if (!Array.isArray(decision.timing_evidence))
    fail('MISSING_TIMING_EVIDENCE', {});
  if (decision.opening_mapping === undefined)
    fail('MISSING_OPENING_MAPPING', {});
  if (out.length) return out; // structure first; every check below reads fields

  if (decision.schema !== SCHEMA)
    fail('WRONG_SCHEMA', { schema: decision.schema });

  if (decision.threshold_games_per_hour !== THRESHOLD_GAMES_PER_HOUR) {
    fail('WRONG_THRESHOLD', {
      found: decision.threshold_games_per_hour,
      frozen: THRESHOLD_GAMES_PER_HOUR,
    });
  }
  if (m.timing_games !== TIMING_GAMES) {
    fail('WRONG_TIMING_GAME_COUNT', {
      found: m.timing_games,
      frozen: TIMING_GAMES,
    });
  }
  if (![P_IF_AT_OR_ABOVE, P_IF_BELOW].includes(decision.selected_p)) {
    fail('P_NOT_A_PERMITTED_VALUE', { selected_p: decision.selected_p });
  }

  // --- re-derive, never trust ----------------------------------------------
  let throughput = null;
  try {
    throughput = computeThroughput(m.total_sequential_wall_ms, m.timing_games);
  } catch (err) {
    fail('BAD_WALL_TIME', { message: err.message });
  }
  if (throughput !== null) {
    if (Math.abs(throughput - m.games_per_hour) > 1e-6) {
      fail('THROUGHPUT_NOT_DERIVED_FROM_WALL_TIME', {
        recorded: m.games_per_hour,
        derived: throughput,
      });
    }
    if (deriveP(throughput) !== decision.selected_p) {
      fail('P_NOT_DERIVED_FROM_THROUGHPUT', {
        recorded: decision.selected_p,
        derived: deriveP(throughput),
      });
    }
  }

  if (
    JSON.stringify(decision.opening_mapping) !==
    JSON.stringify(TIMING_OPENING_MAPPING)
  ) {
    fail('OPENING_MAPPING_NOT_FROZEN', { found: decision.opening_mapping });
  }

  // --- timing evidence: exact filenames, and distinct digests ---------------
  const expectedFiles = expectedTimingFilenames();
  const files = [];
  const digests = new Set();
  for (const e of decision.timing_evidence) {
    if (!typeOk(e?.file, 'string') || !/^[0-9a-f]{64}$/.test(e?.sha256 ?? '')) {
      fail('MALFORMED_TIMING_EVIDENCE', { entry: e ?? null });
      continue;
    }
    files.push(e.file);
    // Two DIFFERENT files may not carry the same digest: ten records of one
    // game would otherwise satisfy a filename-only check.
    if (digests.has(e.sha256))
      fail('DUPLICATE_TIMING_DIGEST', { sha256: e.sha256 });
    digests.add(e.sha256);
  }
  if (
    JSON.stringify([...files].sort()) !==
    JSON.stringify([...expectedFiles].sort())
  ) {
    fail('TIMING_EVIDENCE_FILENAMES', {
      found: files,
      expected: expectedFiles,
    });
  }

  // --- bindings -------------------------------------------------------------
  if (poolSha256 !== undefined && decision.opening_pool_sha256 !== poolSha256) {
    fail('POOL_HASH_MISMATCH', {
      decision: decision.opening_pool_sha256,
      actual: poolSha256,
    });
  }
  if (
    surfaceSha256 !== undefined &&
    decision.execution_surface_sha256 !== surfaceSha256
  ) {
    fail('EXECUTION_SURFACE_MISMATCH', {
      decision: decision.execution_surface_sha256,
      actual: surfaceSha256,
    });
  }
  for (const [field, value] of Object.entries(expected)) {
    if (decision[field] !== value) {
      fail('BINDING_MISMATCH', {
        field,
        decision: decision[field],
        expected: value,
      });
    }
  }
  return out;
}

/**
 * Load the decision AS COMMITTED at `commit`, and enforce every binding.
 *
 * There is deliberately no option meaning "no decision, carry on", and no way
 * to relax the commitment requirement: both would restore exactly the mutable
 * binding this replaces.
 */
export async function loadCommittedDecision({
  commit = 'HEAD',
  relPath = P_DECISION_RELPATH,
  repoRoot = REPO_ROOT,
  poolRelPath = POOL_RELPATH,
  expected = {},
} = {}) {
  const bytes = readCommittedBlob(relPath, commit, repoRoot); // throws NOT_COMMITTED

  let decision;
  try {
    decision = JSON.parse(bytes.toString('utf8'));
  } catch (err) {
    throw new PDecisionError(
      'DECISION_UNPARSEABLE',
      `${relPath} at ${commit}: ${err.message}`
    );
  }

  // The pool and the execution surface are read from the SAME commit, so the
  // decision is checked against the code and data that commit actually holds.
  const poolSha256 = sha256(readCommittedBlob(poolRelPath, commit, repoRoot));
  const surfaceSha256 = executionSurfaceDigest(commit, repoRoot);

  const failures = decisionFailures(decision, {
    poolSha256,
    surfaceSha256,
    expected,
  });
  if (failures.length) {
    throw new PDecisionError(
      'DECISION_INVALID',
      `${relPath} at ${commit} is not a valid P decision: ${failures
        .map((f) => f.code)
        .join(', ')}`
    );
  }
  return decision;
}

/** Build a decision from a completed timing run. Every derived value computed here. */
export function buildDecision({
  totalSequentialWallMs,
  timingEvidence,
  openingPoolSha256,
  executionCommit,
  executionSurfaceSha256,
  baselineModelId,
  candidateModelId,
  ortVersion,
  ortConfig,
  nSimulations,
  cPuct,
  moveTemp,
}) {
  const gamesPerHour = computeThroughput(totalSequentialWallMs, TIMING_GAMES);
  return {
    schema: SCHEMA,
    specification:
      'docs/superpowers/2026-08-14-product-stack-comparison-specification.md',
    selected_p: deriveP(gamesPerHour),
    threshold_games_per_hour: THRESHOLD_GAMES_PER_HOUR,
    derivation: `games_per_hour >= ${THRESHOLD_GAMES_PER_HOUR} -> P=${P_IF_AT_OR_ABOVE}, else P=${P_IF_BELOW}`,
    measured: {
      timing_games: TIMING_GAMES,
      total_sequential_wall_ms: totalSequentialWallMs,
      games_per_hour: gamesPerHour,
    },
    measurement_note:
      'One monotonic wall-clock span, started immediately before game 1 first MCTS search and stopped immediately after game 10 atomic rename. Evidence digests are computed after the clock stops. Both sessions were loaded and both contracts asserted before it started.',
    opening_pool_sha256: openingPoolSha256,
    opening_mapping: TIMING_OPENING_MAPPING,
    timing_evidence: timingEvidence,
    execution_commit: executionCommit,
    execution_surface_sha256: executionSurfaceSha256,
    execution_surface_files: [...EXECUTION_SURFACE_FILES],
    baseline_model_id: baselineModelId,
    candidate_model_id: candidateModelId,
    ort_version: ortVersion,
    ort_config: ortConfig,
    n_simulations: nSimulations,
    c_puct: cPuct,
    move_temp: moveTemp,
    execution_mode: 'one process, sequential, no concurrency',
    outcome_blind_note:
      'Both timing arms are self-play, so the measurement carries no comparative information and P cannot have been chosen with knowledge of the matchup.',
  };
}

/** Convenience for callers that must read the committed pool at a commit. */
export const committedPoolSha256 = (commit = 'HEAD', repoRoot = REPO_ROOT) =>
  sha256(readCommittedBlob(POOL_RELPATH, commit, repoRoot));
