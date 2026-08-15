#!/usr/bin/env node
/**
 * Independent analyser for a product-stack match.
 *
 *   node tests/product_match/analyse.mjs <run_dir> <openings.json> <out.json>
 *
 * Deliberately shares NO code with `harness.mjs`. It imports only `TwixtState`
 * and the standard library, and re-derives every quantity it could otherwise
 * take on trust — so the process that produced the evidence cannot also decide
 * what the evidence says.
 *
 * Every threshold is transcribed from the specification; none is computed from
 * the data.
 *
 * Specification: docs/superpowers/2026-08-14-product-stack-comparison-specification.md
 * §6 (statistic), §9 (integrity), §10 (schema and acceptance).
 */
import { readFile, writeFile, readdir, mkdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, join, resolve } from 'node:path';

import { TwixtState, MAX_PLIES } from '../../server/gameLogic.js';

/**
 * The frozen constants. Defaults are the specification's; tests may pass their
 * own `spec`, which makes any relaxation explicit and visible at the call site
 * rather than hidden behind a flag.
 */
export const FROZEN_SPEC = {
  schema: 'twixt-product-match/1',
  baselineModelId: '1d64027db521a50f',
  candidateModelId: 'c34b7ff3297c785a',
  nSimulations: 800,
  cPuct: 1.5,
  moveTemp: 0,
  ortConfig: 'no options supplied',
  // §6.1 — only two P values are reachable, so both criticals are frozen.
  // Verified independently by Cornish-Fisher expansion to all ten digits.
  tCritical: { 100: 1.9842169515, 200: 1.9719565442 },
  bootstrapReplicates: 10000,
  bootstrapSeed: 20260814,
  bootstrapLowerIndex: 250,
  bootstrapUpperIndex: 9749,
};

const sha256 = (s) => createHash('sha256').update(s).digest('hex');

/**
 * Every §10 field, with its type. Checked structurally BEFORE any semantic
 * comparison: an absent field compares `undefined === undefined` and passes,
 * so equality checks alone would accept evidence that simply omits the thing
 * being checked.
 */
const REQUIRED_FIELDS = {
  kind: 'string',
  schema: 'string',
  opening_id: 'integer',
  opening_sha256: 'string',
  pair_index: 'integer',
  game_in_pair: 'integer',
  baseline_model_id: 'string',
  candidate_model_id: 'string',
  red_model_id: 'string',
  black_model_id: 'string',
  moves: 'moves',
  result: 'string',
  candidate_score: 'number',
  termination: 'string',
  ply_count: 'integer',
  n_simulations: 'integer',
  c_puct: 'number',
  move_temp: 'number',
  ort_version: 'string',
  ort_config: 'string',
  execution_commit: 'string',
  elapsed_ms: 'number',
};

/** The run fingerprint's fields and their types. All mandatory. */
const FINGERPRINT_TYPES = {
  execution_commit: 'string',
  schema: 'string',
  ort_version: 'string',
  ort_config: 'string',
  n_simulations: 'integer',
  c_puct: 'number',
  move_temp: 'number',
  baseline_model_id: 'string',
  candidate_model_id: 'string',
};

const RESULTS = new Set(['red', 'black', 'draw']);
const TERMINATIONS = new Set(['win', 'no_legal_moves', 'max_plies']);

const typeOk = (value, type) => {
  if (value === undefined || value === null) return false;
  if (type === 'string') return typeof value === 'string' && value.length > 0;
  if (type === 'number')
    return typeof value === 'number' && Number.isFinite(value);
  if (type === 'integer') return Number.isInteger(value);
  if (type === 'moves') {
    return (
      Array.isArray(value) &&
      value.length > 0 &&
      value.every(
        (m) =>
          Array.isArray(m) &&
          m.length === 2 &&
          m.every((x) => Number.isInteger(x))
      )
    );
  }
  return false;
};

/**
 * Structural validation of the run's own committed metadata.
 *
 * The fingerprint is MANDATORY. Treating it as optional would mean deleting it
 * disables the binding it exists to enforce — a check that can be switched off
 * by removing the thing being checked is not a check.
 */
export function runMetaFailures(runMeta, spec) {
  const out = [];
  if (
    runMeta === null ||
    typeof runMeta !== 'object' ||
    Array.isArray(runMeta)
  ) {
    return [{ code: 'RUN_METADATA_NOT_AN_OBJECT', detail: {} }];
  }
  if (!Number.isInteger(runMeta.P) || runMeta.P < 2) {
    out.push({ code: 'BAD_COMMITTED_P', detail: { P: runMeta.P ?? null } });
  } else if (spec.tCritical[runMeta.P] === undefined) {
    // Rejected up front, before any replay: an unsupported P means this is not
    // the run that was preregistered, whatever its evidence looks like.
    out.push({
      code: 'UNSUPPORTED_P',
      detail: {
        P: runMeta.P,
        supported: Object.keys(spec.tCritical).map(Number),
      },
    });
  }
  const fp = runMeta.fingerprint;
  if (fp === null || typeof fp !== 'object' || Array.isArray(fp)) {
    out.push({ code: 'MISSING_RUN_FINGERPRINT', detail: {} });
    return out;
  }
  for (const [field, type] of Object.entries(FINGERPRINT_TYPES)) {
    if (!typeOk(fp[field], type)) {
      out.push({
        code: 'MALFORMED_RUN_FINGERPRINT',
        detail: { field, expected: type, found: fp[field] ?? null },
      });
    }
  }
  const extra = Object.keys(fp).filter((k) => !(k in FINGERPRINT_TYPES));
  if (extra.length)
    out.push({ code: 'UNEXPECTED_FINGERPRINT_FIELDS', detail: { extra } });
  return out;
}

/** Structural validation of one sidecar. Returns a list of failures. */
export function structuralFailures(sidecar, where) {
  const out = [];
  if (
    sidecar === null ||
    typeof sidecar !== 'object' ||
    Array.isArray(sidecar)
  ) {
    return [{ code: 'SIDECAR_NOT_AN_OBJECT', detail: { where } }];
  }
  for (const [field, type] of Object.entries(REQUIRED_FIELDS)) {
    if (!typeOk(sidecar[field], type)) {
      out.push({
        code: 'MISSING_OR_MALFORMED_FIELD',
        detail: { where, field, expected: type, found: sidecar[field] ?? null },
      });
    }
  }
  const extra = Object.keys(sidecar).filter((k) => !(k in REQUIRED_FIELDS));
  if (extra.length)
    out.push({ code: 'UNEXPECTED_FIELDS', detail: { where, extra } });

  if (sidecar.result !== undefined && !RESULTS.has(sidecar.result))
    out.push({
      code: 'BAD_RESULT_VALUE',
      detail: { where, result: sidecar.result },
    });
  if (
    sidecar.termination !== undefined &&
    !TERMINATIONS.has(sidecar.termination)
  )
    out.push({
      code: 'BAD_TERMINATION_VALUE',
      detail: { where, termination: sidecar.termination },
    });
  if (![0, 1].includes(sidecar.game_in_pair))
    out.push({
      code: 'BAD_GAME_IN_PAIR_VALUE',
      detail: { where, value: sidecar.game_in_pair },
    });
  if (![0, 0.5, 1].includes(sidecar.candidate_score))
    out.push({
      code: 'BAD_CANDIDATE_SCORE_VALUE',
      detail: { where, value: sidecar.candidate_score },
    });
  return out;
}

// --- statistics -------------------------------------------------------------

/** mulberry32 — the same generator the corpus and openings use. */
export function mulberry32(seed) {
  let a = seed | 0;
  return function next() {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * §6.1 percentile bootstrap, fully determined.
 *
 * One continuous RNG stream seeded once; replicate `b` draws `P` indices in
 * order; endpoints are literal order statistics of the ascending-sorted
 * replicate means, with no interpolation and no library quantile convention.
 */
export function bootstrapInterval(pairScores, spec = FROZEN_SPEC) {
  const P = pairScores.length;
  const rand = mulberry32(spec.bootstrapSeed);
  const means = new Array(spec.bootstrapReplicates);
  for (let b = 0; b < spec.bootstrapReplicates; b++) {
    let sum = 0;
    for (let i = 0; i < P; i++) {
      const idx = Math.min(P - 1, Math.floor(rand() * P));
      sum += pairScores[idx];
    }
    means[b] = sum / P;
  }
  means.sort((x, y) => x - y);
  return {
    lower: means[spec.bootstrapLowerIndex],
    upper: means[spec.bootstrapUpperIndex],
  };
}

/** §6.1 Student-t cross-check, using the frozen critical value for this P. */
export function tInterval(pairScores, spec = FROZEN_SPEC) {
  const P = pairScores.length;
  const mean = pairScores.reduce((s, v) => s + v, 0) / P;
  const variance =
    pairScores.reduce((s, v) => s + (v - mean) ** 2, 0) / (P - 1);
  const sd = Math.sqrt(variance);
  const tCrit = spec.tCritical[P];
  if (tCrit === undefined) {
    throw new Error(
      `no frozen t critical value for P=${P}; the specification allows only ${Object.keys(spec.tCritical).join(', ')}`
    );
  }
  const half = tCrit * (sd / Math.sqrt(P));
  return { mean, sd, lower: mean - half, upper: mean + half };
}

/**
 * Where one interval sits relative to parity.
 *
 * Three-valued on purpose: comparing only lower bounds would call "weaker" and
 * "inconclusive" an agreement, because neither lower bound exceeds 0.5.
 */
export const classify = (ci) =>
  ci.lower > 0.5 ? 'stronger' : ci.upper < 0.5 ? 'weaker' : 'inconclusive';

/** §6.2 — both methods must agree, in either direction. */
export function decide(boot, t) {
  const b = classify(boot);
  const s = classify(t);
  if (b !== s) return 'UNRESOLVED';
  if (b === 'stronger') return 'CANDIDATE_STRONGER';
  if (b === 'weaker') return 'CANDIDATE_WEAKER';
  return 'UNRESOLVED';
}

// --- analysis ---------------------------------------------------------------

export async function analyse(runDir, openings, spec = FROZEN_SPEC) {
  const failures = [];
  const fail = (code, detail) => failures.push({ code, detail });

  // `P` is the run's PREREGISTERED size, read from its own committed metadata --
  // never inferred from how many sidecars happen to be on disk. Inferring it
  // would let the first 100 finished pairs of a P=200 run be analysed as a
  // complete P=100 result: an interim peek wearing a final verdict.
  let runMeta;
  try {
    runMeta = JSON.parse(await readFile(join(runDir, 'run.json'), 'utf8'));
  } catch {
    return {
      verdict: 'REJECTED',
      failures: [{ code: 'NO_RUN_METADATA', detail: join(runDir, 'run.json') }],
    };
  }
  const metaFailures = runMetaFailures(runMeta, spec);
  if (metaFailures.length)
    return { verdict: 'REJECTED', failures: metaFailures };
  const P = runMeta.P;

  const matchDir = join(runDir, 'match');
  let names;
  try {
    names = (await readdir(matchDir)).filter((n) => n.endsWith('.json'));
  } catch {
    return {
      verdict: 'REJECTED',
      failures: [{ code: 'NO_MATCH_DIRECTORY', detail: matchDir }],
    };
  }
  if (names.length === 0) {
    return {
      verdict: 'REJECTED',
      failures: [{ code: 'EMPTY_MATCH_DIRECTORY', detail: matchDir }],
    };
  }
  if (names.length !== 2 * P) {
    return {
      verdict: 'REJECTED',
      failures: [
        {
          code: 'SIDECAR_COUNT',
          detail: { found: names.length, required: 2 * P, P },
        },
      ],
    };
  }

  const sidecars = [];
  for (const n of names) {
    let parsed;
    try {
      parsed = JSON.parse(await readFile(join(matchDir, n), 'utf8'));
    } catch (err) {
      fail('UNPARSEABLE_SIDECAR', { file: n, message: err.message });
      continue;
    }
    sidecars.push(parsed);
    for (const f of structuralFailures(parsed, n)) failures.push(f);
  }
  // Structure first: every later check compares fields, and a comparison
  // against an ABSENT field succeeds vacuously.
  if (failures.length) return { verdict: 'REJECTED', failures };

  // Only match evidence is admissible. Timing sidecars are self-play by design
  // and live in a separate namespace; a stray one here is a hard reject, not a
  // row to skip.
  for (const s of sidecars) {
    if (s.kind !== 'match')
      fail('NOT_A_MATCH_SIDECAR', { kind: s.kind, pair_index: s.pair_index });
    if (s.schema !== spec.schema) fail('WRONG_SCHEMA', { schema: s.schema });
  }
  if (failures.length) return { verdict: 'REJECTED', failures };

  // --- run fingerprint: one implementation, or no result --------------------
  const FP = [
    'execution_commit',
    'schema',
    'ort_version',
    'ort_config',
    'n_simulations',
    'c_puct',
    'move_temp',
    'baseline_model_id',
    'candidate_model_id',
  ];
  const first = sidecars[0];
  // Unconditional: run.json's fingerprint is validated above, so it is always
  // present and always binding.
  for (const f of FP) {
    if (first[f] !== runMeta.fingerprint[f])
      fail('FINGERPRINT_NOT_THE_COMMITTED_RUN', {
        field: f,
        committed: runMeta.fingerprint[f],
        found: first[f],
      });
  }
  for (const s of sidecars) {
    for (const f of FP) {
      if (s[f] !== first[f]) {
        fail('FINGERPRINT_DRIFT', {
          field: f,
          expected: first[f],
          found: s[f],
          pair_index: s.pair_index,
        });
      }
    }
  }
  // …and the fingerprint must be the configuration the specification names.
  if (first.baseline_model_id !== spec.baselineModelId)
    fail('WRONG_BASELINE_MODEL', { found: first.baseline_model_id });
  if (first.candidate_model_id !== spec.candidateModelId)
    fail('WRONG_CANDIDATE_MODEL', { found: first.candidate_model_id });
  if (first.n_simulations !== spec.nSimulations)
    fail('WRONG_SIMULATIONS', {
      found: first.n_simulations,
      required: spec.nSimulations,
    });
  if (first.c_puct !== spec.cPuct) fail('WRONG_CPUCT', { found: first.c_puct });
  if (first.move_temp !== spec.moveTemp)
    fail('WRONG_MOVE_TEMP', { found: first.move_temp });
  if (first.ort_config !== spec.ortConfig)
    fail('WRONG_ORT_CONFIG', { found: first.ort_config });

  // --- cardinality: TWO sidecars per pair -----------------------------------
  const byPair = new Map();
  for (const s of sidecars) {
    if (!byPair.has(s.pair_index)) byPair.set(s.pair_index, []);
    byPair.get(s.pair_index).push(s);
  }
  if (byPair.size !== P)
    fail('PAIR_COUNT', { pairs: byPair.size, expected: P });

  for (const [pairIndex, games] of byPair) {
    if (games.length !== 2)
      fail('PAIR_NOT_COMPLETE', { pair_index: pairIndex, games: games.length });
    const slots = games.map((g) => g.game_in_pair).sort();
    if (slots.length !== 2 || slots[0] !== 0 || slots[1] !== 1)
      fail('BAD_GAME_IN_PAIR', { pair_index: pairIndex, slots });
    if (new Set(games.map((g) => g.opening_id)).size !== 1)
      fail('PAIR_OPENING_MISMATCH', { pair_index: pairIndex });
  }

  // The opening set must be the frozen PREFIX, not merely P unique values:
  // `P` is chosen from timing alone, so which openings it selects has to follow
  // mechanically from `P`.
  const openingIds = [...byPair.keys()].sort((a, b) => a - b);
  for (const s of sidecars) {
    if (s.opening_id !== s.pair_index)
      fail('OPENING_ID_NOT_PAIR_INDEX', {
        pair_index: s.pair_index,
        opening_id: s.opening_id,
      });
  }
  const expectedIds = Array.from({ length: P }, (_, i) => i);
  if (JSON.stringify(openingIds) !== JSON.stringify(expectedIds))
    fail('OPENING_SET_NOT_PREFIX', {
      found: openingIds,
      expected: `0…${P - 1}`,
    });

  // --- per-game: re-derive everything ---------------------------------------
  const pairScores = new Map();
  for (const s of sidecars) {
    const ctx = { pair_index: s.pair_index, game_in_pair: s.game_in_pair };

    // Colour assignment: the SET must be the two roles, and game_in_pair must
    // determine which is which.
    const colourSet = [s.red_model_id, s.black_model_id].sort();
    const roleSet = [s.baseline_model_id, s.candidate_model_id].sort();
    if (JSON.stringify(colourSet) !== JSON.stringify(roleSet))
      fail('COLOUR_SET_MISMATCH', { ...ctx, colourSet, roleSet });
    const expectRed =
      s.game_in_pair === 0 ? s.candidate_model_id : s.baseline_model_id;
    if (s.red_model_id !== expectRed)
      fail('COLOUR_ASSIGNMENT', {
        ...ctx,
        red_model_id: s.red_model_id,
        expected: expectRed,
      });

    // Opening prefix and hash, against the committed pool.
    const opening = openings[s.opening_id];
    if (!opening) {
      fail('UNKNOWN_OPENING', ctx);
      continue;
    }
    if (s.opening_sha256 !== sha256(JSON.stringify(opening)))
      fail('OPENING_HASH_MISMATCH', ctx);
    const prefix = s.moves.slice(0, opening.length);
    if (JSON.stringify(prefix) !== JSON.stringify(opening))
      fail('OPENING_PREFIX_MISMATCH', ctx);

    // Replay every move, asserting legality at the moment it was played.
    let state = new TwixtState({});
    let illegal = null;
    for (let i = 0; i < s.moves.length; i++) {
      if (state.isTerminal()) {
        illegal = { index: i, reason: 'game_already_terminal' };
        break;
      }
      const m = s.moves[i];
      if (!state.legalMoves().some((x) => x[0] === m[0] && x[1] === m[1])) {
        illegal = { index: i, move: m, reason: 'illegal_move' };
        break;
      }
      state = state.applyMove(m);
    }
    if (illegal) {
      fail('REPLAY_ILLEGAL', { ...ctx, ...illegal });
      continue;
    }
    if (!state.isTerminal()) {
      fail('REPLAY_NOT_TERMINAL', { ...ctx, ply: state.ply });
      continue;
    }

    // Re-derive the recorded facts rather than believing them.
    const result = state.gameResult();
    const termination =
      state.winner() !== null
        ? 'win'
        : state.ply >= MAX_PLIES
          ? 'max_plies'
          : 'no_legal_moves';
    if (result !== s.result)
      fail('RESULT_MISMATCH', { ...ctx, stored: s.result, derived: result });
    if (termination !== s.termination)
      fail('TERMINATION_MISMATCH', {
        ...ctx,
        stored: s.termination,
        derived: termination,
      });
    if (state.ply !== s.ply_count)
      fail('PLY_COUNT_MISMATCH', {
        ...ctx,
        stored: s.ply_count,
        derived: state.ply,
      });

    // candidate_score is stored for legibility only; recompute it from the
    // re-derived result, the role fields and the colour assignment. The whole
    // comparison rests on this number.
    const candidateColour =
      s.red_model_id === s.candidate_model_id ? 'red' : 'black';
    const derivedScore =
      result === 'draw' ? 0.5 : result === candidateColour ? 1.0 : 0.0;
    if (derivedScore !== s.candidate_score)
      fail('CANDIDATE_SCORE_MISMATCH', {
        ...ctx,
        stored: s.candidate_score,
        derived: derivedScore,
      });

    if (!pairScores.has(s.pair_index)) pairScores.set(s.pair_index, []);
    pairScores.get(s.pair_index).push(derivedScore);
  }

  if (failures.length) return { verdict: 'REJECTED', failures };

  // --- statistic ------------------------------------------------------------
  const scores = [...pairScores.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, gs]) => (gs[0] + gs[1]) / 2);

  const boot = bootstrapInterval(scores, spec);
  const t = tInterval(scores, spec);
  const decision = decide(boot, t);

  const tally = { win: 0, draw: 0, loss: 0 };
  for (const s of scores) {
    if (s > 0.5) tally.win += 1;
    else if (s === 0.5) tally.draw += 1;
    else tally.loss += 1;
  }

  return {
    schema: 'twixt-product-match-result/1',
    verdict: 'ACCEPTED',
    decision,
    P,
    mean_pair_score: t.mean,
    sd_pair_score: t.sd,
    bootstrap: boot,
    t_interval: { lower: t.lower, upper: t.upper },
    bootstrap_class: classify(boot),
    t_class: classify(t),
    // Compares three-valued classifications, not lower bounds: "weaker" and
    // "inconclusive" both have a lower bound below 0.5, so a lower-bound
    // comparison would report them as agreeing.
    methods_agree: classify(boot) === classify(t),
    pair_tally: tally,
    pair_scores: scores,
    fingerprint: Object.fromEntries(FP.map((f) => [f, first[f]])),
    failures: [],
  };
}

// --- CLI --------------------------------------------------------------------

const isMain =
  process.argv[1] && import.meta.url === `file://${process.argv[1]}`;
if (isMain) {
  const [runDir, openingsPath, outPath] = process.argv.slice(2);
  if (!runDir || !openingsPath || !outPath) {
    console.error('usage: analyse.mjs <run_dir> <openings.json> <out.json>');
    process.exit(2);
  }
  const openingsFile = JSON.parse(
    await readFile(resolve(openingsPath), 'utf8')
  );
  const openings = openingsFile.openings ?? openingsFile;
  const report = await analyse(resolve(runDir), openings);
  await mkdir(dirname(resolve(outPath)), { recursive: true });
  await writeFile(resolve(outPath), JSON.stringify(report, null, 2));

  if (report.verdict === 'REJECTED') {
    console.log(`\nANALYSIS REJECTED — ${report.failures.length} failure(s)\n`);
    for (const f of report.failures.slice(0, 20)) console.log(`  ${f.code}`);
    process.exit(1);
  }
  console.log(`\n${report.decision}   P=${report.P}`);
  console.log(
    `  mean pair score ${report.mean_pair_score.toFixed(4)}  (sd ${report.sd_pair_score.toFixed(4)})`
  );
  console.log(
    `  bootstrap 95%   [${report.bootstrap.lower.toFixed(4)}, ${report.bootstrap.upper.toFixed(4)}]`
  );
  console.log(
    `  t 95%           [${report.t_interval.lower.toFixed(4)}, ${report.t_interval.upper.toFixed(4)}]`
  );
  console.log(
    `  pairs W/D/L     ${report.pair_tally.win}/${report.pair_tally.draw}/${report.pair_tally.loss}`
  );
  process.exit(0);
}
