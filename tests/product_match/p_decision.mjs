#!/usr/bin/env node
/**
 * The committed P-decision artifact: schema, derivation, and validator.
 *
 * `P` is chosen from TIMING ALONE and must be fixed before the first match game
 * is played. Binding it to a committed artifact — rather than to `run.json`,
 * which is mutable runtime metadata a match writes about itself — is what stops
 * the size of the sample being decided, or revised, by anything the match
 * itself observed.
 *
 * Deliberately a standalone module. The analyser must enforce this binding
 * without importing the harness, so nothing here loads a model, plays a game,
 * or imports either of those.
 *
 * The artifact does not exist yet and is not created by this file. It is
 * written by the timing runner after the ten timing games, then committed. No
 * placeholder is provided: a file that looked like a decision but was not one
 * is worse than none at all.
 *
 * Specification: docs/superpowers/2026-08-14-product-stack-comparison-specification.md §7.3
 */
import { readFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
export const REPO_ROOT = join(HERE, '..', '..');

/** The one location a decision may live. Not configurable. */
export const P_DECISION_RELPATH = 'tests/product_match/p_decision.json';
export const P_DECISION_PATH = join(REPO_ROOT, P_DECISION_RELPATH);

export const SCHEMA = 'twixt-p-decision/1';

/** Frozen by specification §7.3. */
export const THRESHOLD_GAMES_PER_HOUR = 8.8;
export const TIMING_GAMES = 10;
export const P_IF_AT_OR_ABOVE = 200;
export const P_IF_BELOW = 100;

/**
 * The exact opening mapping, frozen so the timing sample cannot be reshaped.
 *
 * Self-play on both sides is what makes the smoke OUTCOME-BLIND: a model
 * playing itself yields no comparative information, so `P` cannot be chosen
 * with any knowledge of the matchup.
 */
export const TIMING_OPENING_MAPPING = Object.freeze({
  baseline_self_play: Object.freeze([200, 201, 202, 203, 204]),
  candidate_self_play: Object.freeze([205, 206, 207, 208, 209]),
});

export class PDecisionError extends Error {
  constructor(code, message) {
    super(message);
    this.name = 'PDecisionError';
    this.code = code;
  }
}

export const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');

/**
 * Throughput, exactly as §7.3 defines it: ONE wall-clock span, never a sum of
 * per-game elapsed times, which would drop inter-game overhead and could land
 * on the other side of the threshold.
 */
export function computeThroughput(
  totalSequentialWallMs,
  nGames = TIMING_GAMES
) {
  if (!Number.isFinite(totalSequentialWallMs) || totalSequentialWallMs <= 0) {
    throw new PDecisionError(
      'BAD_WALL_TIME',
      `total_sequential_wall_ms must be positive`
    );
  }
  return (nGames * 3_600_000) / totalSequentialWallMs;
}

/** The whole of the choice: `≥ 8.8 → 200`, `< 8.8 → 100`. Nothing else. */
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
 * Everything derivable is RE-DERIVED. A decision that merely agrees with itself
 * proves nothing: the recorded `selected_p` and `games_per_hour` are exactly
 * the numbers someone would have to fake, so they are recomputed from the
 * measured wall time and compared.
 */
export function decisionFailures(decision, { poolSha256, expected = {} } = {}) {
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
  if (!Array.isArray(decision.timing_evidence)) {
    fail('MISSING_TIMING_EVIDENCE', {});
  }
  if (decision.opening_mapping === undefined)
    fail('MISSING_OPENING_MAPPING', {});
  if (out.length) return out; // structure first; every check below reads fields

  if (decision.schema !== SCHEMA)
    fail('WRONG_SCHEMA', { schema: decision.schema });

  // --- the frozen constants may not be restated differently -----------------
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
    const derivedP = deriveP(throughput);
    if (derivedP !== decision.selected_p) {
      fail('P_NOT_DERIVED_FROM_THROUGHPUT', {
        recorded: decision.selected_p,
        derived: derivedP,
        games_per_hour: throughput,
      });
    }
  }

  // --- the opening mapping is frozen ---------------------------------------
  if (
    JSON.stringify(decision.opening_mapping) !==
    JSON.stringify(TIMING_OPENING_MAPPING)
  ) {
    fail('OPENING_MAPPING_NOT_FROZEN', { found: decision.opening_mapping });
  }

  // --- timing evidence ------------------------------------------------------
  if (decision.timing_evidence.length !== TIMING_GAMES) {
    fail('TIMING_EVIDENCE_COUNT', {
      found: decision.timing_evidence.length,
      required: TIMING_GAMES,
    });
  }
  const seenFiles = new Set();
  for (const e of decision.timing_evidence) {
    if (!typeOk(e?.file, 'string') || !typeOk(e?.sha256, 'string')) {
      fail('MALFORMED_TIMING_EVIDENCE', { entry: e ?? null });
      continue;
    }
    if (!/^[0-9a-f]{64}$/.test(e.sha256))
      fail('MALFORMED_TIMING_EVIDENCE', { file: e.file });
    if (seenFiles.has(e.file))
      fail('DUPLICATE_TIMING_EVIDENCE', { file: e.file });
    seenFiles.add(e.file);
  }

  // --- bindings -------------------------------------------------------------
  if (poolSha256 !== undefined && decision.opening_pool_sha256 !== poolSha256) {
    fail('POOL_HASH_MISMATCH', {
      decision: decision.opening_pool_sha256,
      actual: poolSha256,
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

/** Is the artifact tracked by git? An uncommitted decision is not a decision. */
export function isTracked(relPath, repoRoot = REPO_ROOT) {
  try {
    execFileSync('git', ['ls-files', '--error-unmatch', relPath], {
      cwd: repoRoot,
      stdio: 'ignore',
    });
    return true;
  } catch {
    return false;
  }
}

/**
 * Load and fully validate the committed decision.
 *
 * Throws on ANY problem — missing, untracked, unparseable, malformed or
 * mismatched. There is deliberately no return value meaning "no decision, carry
 * on": the whole point is that a match cannot begin without one, and a
 * fallback to `run.json.P` would restore exactly the mutable binding this
 * replaces.
 */
export async function loadCommittedDecision({
  path = P_DECISION_PATH,
  relPath = P_DECISION_RELPATH,
  repoRoot = REPO_ROOT,
  poolSha256,
  expected,
  requireTracked = true,
} = {}) {
  let text;
  try {
    text = await readFile(path, 'utf8');
  } catch (err) {
    if (err.code === 'ENOENT') {
      throw new PDecisionError(
        'DECISION_MISSING',
        `no committed P decision at ${relative(repoRoot, path)}; the timing smoke must run and its ` +
          `decision be committed before any match game`
      );
    }
    throw new PDecisionError(
      'DECISION_UNREADABLE',
      `cannot read ${path}: ${err.message}`
    );
  }

  if (requireTracked && !isTracked(relPath, repoRoot)) {
    throw new PDecisionError(
      'DECISION_NOT_COMMITTED',
      `${relPath} exists but is not tracked by git; an uncommitted decision could be edited ` +
        `after the match starts, which is the mutability this binding exists to remove`
    );
  }

  let decision;
  try {
    decision = JSON.parse(text);
  } catch (err) {
    throw new PDecisionError(
      'DECISION_UNPARSEABLE',
      `${relPath} is not valid JSON: ${err.message}`
    );
  }

  const failures = decisionFailures(decision, { poolSha256, expected });
  if (failures.length) {
    throw new PDecisionError(
      'DECISION_INVALID',
      `${relPath} is not a valid P decision: ${failures.map((f) => f.code).join(', ')}`
    );
  }
  return decision;
}

/**
 * Build a decision from a completed timing run.
 *
 * Every derived quantity is computed here rather than accepted from a caller,
 * so a decision cannot record a `P` its own measurement does not imply.
 */
export function buildDecision({
  totalSequentialWallMs,
  timingEvidence,
  openingPoolSha256,
  executionCommit,
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
      'One wall-clock span, from immediately before the first MCTS search of game 1 to completion of the atomic rename of game 10 sidecar. Not a sum of per-game elapsed times, which would exclude inter-game overhead. Both sessions were loaded and both contracts asserted before the clock started.',
    opening_pool_sha256: openingPoolSha256,
    opening_mapping: TIMING_OPENING_MAPPING,
    timing_evidence: timingEvidence,
    execution_commit: executionCommit,
    baseline_model_id: baselineModelId,
    candidate_model_id: candidateModelId,
    ort_version: ortVersion,
    ort_config: ortConfig,
    n_simulations: nSimulations,
    c_puct: cPuct,
    move_temp: moveTemp,
    execution_mode: 'one process, sequential, no concurrency',
    outcome_blind_note:
      'Both timing arms are self-play, so the measurement carries no comparative information about the two models and P cannot have been chosen with knowledge of the matchup.',
  };
}
