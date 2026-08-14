#!/usr/bin/env node
/**
 * Product-stack match harness.
 *
 * Plays games between two committed model artifacts using the product's own
 * stack — Node ONNX Runtime, `server/mcts.js`, and the shipped readout policy —
 * and writes one atomic sidecar per game.
 *
 * It measures nothing and decides nothing. Validation and the statistic live in
 * `analyse.mjs`, which shares no code with this file, so the thing that
 * produces the evidence cannot also rule on it.
 *
 * Specification: docs/superpowers/2026-08-14-product-stack-comparison-specification.md
 * §3 (harness), §5.1 (Arm A), §8 (resumption), §10 (schema).
 *
 * This module PLAYS GAMES. Running it against the real opening pool is a
 * separately authorized action; nothing here selects `P` or generates openings.
 */
import { readFile, writeFile, rename, mkdir, readdir } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { AlphaZeroInference } from '../../server/inference.js';
import { MCTS } from '../../server/mcts.js';
import { TwixtState, MAX_PLIES } from '../../server/gameLogic.js';
import {
  resolveModel,
  assertSessionContract,
} from '../../server/model_manifest.js';
import {
  resolvePolicy,
  selectMoveForRequest,
} from '../../server/readout_policy.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(HERE, '..', '..');

export const SCHEMA = 'twixt-product-match/1';

/** The frozen identities from specification §2. */
export const BASELINE_MODEL_ID = '1d64027db521a50f';
export const CANDIDATE_MODEL_ID = 'c34b7ff3297c785a';

const sha256 = (s) => createHash('sha256').update(s).digest('hex');

export class HarnessError extends Error {
  constructor(code, message) {
    super(message);
    this.name = 'HarnessError';
    this.code = code;
  }
}
const fail = (code, msg) => {
  throw new HarnessError(code, msg);
};

/**
 * Load a model exactly as the product's startup path does — TWO calls.
 *
 * `resolveModel` validates the manifest, both file hashes and sizes, and the
 * graph-to-sidecar external-data binding. It returns before any session exists
 * and therefore cannot check the tensor contract; `assertSessionContract` does
 * that separately, after `load()`, mirroring `server/index.js`. A harness that
 * stopped at the first call would validate more weakly than the server it is
 * measuring.
 */
export async function loadModel(modelDir) {
  const { manifest, graphPath } = await resolveModel({
    MODEL_MANIFEST: join(modelDir, 'manifest.json'),
  });
  const inference = new AlphaZeroInference(graphPath);
  await inference.load();
  assertSessionContract(manifest, inference.session, inference.maxMoves);
  return { manifest, inference, modelId: manifest.model_id };
}

/** `onnxruntime-node` version, recorded in every sidecar. */
export async function ortVersion() {
  const pkg = JSON.parse(
    await readFile(
      join(REPO_ROOT, 'node_modules/onnxruntime-node/package.json'),
      'utf8'
    )
  );
  return pkg.version;
}

/**
 * Git HEAD, with the worktree asserted clean.
 *
 * A dirty worktree means the recorded commit does not describe the code that
 * actually ran, which would silently break the §10 run fingerprint.
 */
export function executionCommit({ requireClean = true } = {}) {
  const git = (...args) =>
    execFileSync('git', args, { cwd: REPO_ROOT }).toString().trim();
  const commit = git('rev-parse', 'HEAD');
  if (requireClean && git('status', '--porcelain') !== '') {
    fail(
      'DIRTY_WORKTREE',
      'worktree has uncommitted changes; execution_commit would not describe the running code'
    );
  }
  return commit;
}

/**
 * Play one game to its natural end.
 *
 * Termination is `state.isTerminal()` ONLY — win, no legal moves, or the
 * product's own `MAX_PLIES` forced draw. No harness-imposed cap: `MAX_PLIES`
 * also feeds tensor channel 23, so a shorter cap would both cut the game short
 * and feed the network a different phase signal.
 */
export async function playGame({
  redInference,
  blackInference,
  openingMoves,
  nSimulations,
  cPuct = 1.5,
  difficulty = 'hard',
}) {
  // The readout temperature is taken from the shipped table, never passed in.
  // Accepting it as a parameter would let a caller play at a different
  // temperature while still labelling the game 'hard' — the readout is the
  // single most drift-prone thing here, so the product owns it outright.
  const policy = resolvePolicy({ difficulty });
  const nSims = nSimulations ?? policy.nSims;

  const redMcts = new MCTS(redInference, { nSimulations: nSims, cPuct });
  const blackMcts = new MCTS(blackInference, { nSimulations: nSims, cPuct });

  let state = new TwixtState({});
  const moves = [];
  for (const m of openingMoves) {
    const legal = state.legalMoves();
    if (!legal.some((x) => x[0] === m[0] && x[1] === m[1])) {
      fail(
        'ILLEGAL_OPENING',
        `opening move [${m}] is not legal at ply ${moves.length}`
      );
    }
    state = state.applyMove(m);
    moves.push([m[0], m[1]]);
  }

  const t0 = Date.now();
  while (!state.isTerminal()) {
    const mcts = state.toMove === 'red' ? redMcts : blackMcts;
    const { visitCounts } = await mcts.search(state);
    if (visitCounts.size === 0)
      fail('EMPTY_SEARCH', `search returned no moves at ply ${state.ply}`);
    // The shipped readout seam, so the harness cannot drift from the product.
    const { moveKey } = selectMoveForRequest({
      visitCounts,
      difficulty,
      selectMove: (counts, temp) => mcts.selectMove(counts, temp),
    });
    const move = moveKey.split(',').map(Number);
    state = state.applyMove(move);
    moves.push([move[0], move[1]]);
  }
  const elapsedMs = Date.now() - t0;

  const result = state.gameResult();
  const termination =
    state.winner() !== null
      ? 'win'
      : state.ply >= MAX_PLIES
        ? 'max_plies'
        : 'no_legal_moves';

  return {
    moves,
    result,
    termination,
    plyCount: state.ply,
    elapsedMs,
    nSimulations: nSims,
    moveTemp: policy.moveTemp,
  };
}

/** The shipped hard-difficulty policy, so callers need not restate it. */
export const hardPolicy = () => resolvePolicy({ difficulty: 'hard' });

/**
 * Build a §10 sidecar.
 *
 * `candidate_score` is stored for legibility. The analyser recomputes it and
 * requires equality; it is never consumed on trust.
 */
export function buildSidecar({
  kind = 'match',
  openingId,
  openingMoves,
  pairIndex,
  gameInPair,
  baselineModelId,
  candidateModelId,
  game,
  nSimulations,
  cPuct,
  moveTemp,
  ortVersion: ortV,
  executionCommit: commit,
}) {
  // game_in_pair fixes the colour assignment: 0 = candidate red, 1 = candidate black.
  const candidateIsRed = gameInPair === 0;
  const redModelId = candidateIsRed ? candidateModelId : baselineModelId;
  const blackModelId = candidateIsRed ? baselineModelId : candidateModelId;

  const candidateColour = candidateIsRed ? 'red' : 'black';
  const candidateScore =
    game.result === 'draw' ? 0.5 : game.result === candidateColour ? 1.0 : 0.0;

  return {
    kind,
    schema: SCHEMA,
    opening_id: openingId,
    opening_sha256: sha256(JSON.stringify(openingMoves)),
    pair_index: pairIndex,
    game_in_pair: gameInPair,
    baseline_model_id: baselineModelId,
    candidate_model_id: candidateModelId,
    red_model_id: redModelId,
    black_model_id: blackModelId,
    moves: game.moves,
    result: game.result,
    candidate_score: candidateScore,
    termination: game.termination,
    ply_count: game.plyCount,
    n_simulations: nSimulations,
    c_puct: cPuct,
    move_temp: moveTemp,
    ort_version: ortV,
    ort_config: 'no options supplied',
    execution_commit: commit,
    elapsed_ms: game.elapsedMs,
  };
}

/** Write via temp file + rename, so a sidecar is never partially observed. */
export async function writeSidecarAtomic(path, obj) {
  const tmp = `${path}.tmp`;
  await writeFile(tmp, JSON.stringify(obj, null, 2));
  await rename(tmp, path);
}

export const sidecarName = (pairIndex, gameInPair) =>
  `pair_${String(pairIndex).padStart(4, '0')}_game_${gameInPair}.json`;

/** The §10 fingerprint fields. Colour-independent by construction. */
export const FINGERPRINT_FIELDS = [
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

export const fingerprintOf = (o) =>
  Object.fromEntries(FINGERPRINT_FIELDS.map((f) => [f, o[f]]));

const sameFingerprint = (a, b) =>
  FINGERPRINT_FIELDS.every((f) => a[f] === b[f]);

/** Fields a resumed replay must reproduce exactly. */
const REPLAY_FIELDS = ['moves', 'result', 'termination', 'ply_count'];

/**
 * Run (or resume) a match over pairs `0…P-1`.
 *
 * Resumption is safe because hard play is deterministic: a resumed run produces
 * exactly what an uninterrupted one would. That is also why a replay MISMATCH
 * is fatal rather than a retry — it would mean determinism does not hold, the
 * most important defect this run could surface.
 *
 * `onGameComplete` exists so tests can interrupt mid-pair. Production passes
 * nothing.
 */
export async function runMatch({
  runDir,
  P,
  openings,
  baselineDir,
  candidateDir,
  nSimulations = null,
  cPuct = 1.5,
  requireCleanWorktree = true,
  onGameComplete = null,
}) {
  const matchDir = join(runDir, 'match');
  const quarantineDir = join(runDir, 'quarantine');
  await mkdir(matchDir, { recursive: true });
  await mkdir(quarantineDir, { recursive: true });

  const baseline = await loadModel(baselineDir);
  const candidate = await loadModel(candidateDir);
  const ortV = await ortVersion();
  const commit = executionCommit({ requireClean: requireCleanWorktree });
  const policy = hardPolicy();
  const nSims = nSimulations ?? policy.nSims;

  const fingerprint = {
    execution_commit: commit,
    schema: SCHEMA,
    ort_version: ortV,
    ort_config: 'no options supplied',
    n_simulations: nSims,
    c_puct: cPuct,
    move_temp: policy.moveTemp,
    baseline_model_id: baseline.modelId,
    candidate_model_id: candidate.modelId,
  };

  // The fingerprint is written once and re-asserted at every start. A clean
  // restart at a different commit is a NEW run, not a resume: finishing a match
  // whose halves were played by different code is the failure this prevents,
  // and it looks entirely benign from the inside.
  const runFile = join(runDir, 'run.json');
  let prior = null;
  try {
    prior = JSON.parse(await readFile(runFile, 'utf8'));
  } catch (err) {
    if (err.code !== 'ENOENT') throw err;
  }
  if (prior) {
    if (!sameFingerprint(prior.fingerprint, fingerprint)) {
      fail(
        'FINGERPRINT_MISMATCH',
        `run fingerprint differs from ${runFile}; a restart under different code or configuration ` +
          `is a new run, and prior pairs may not be reused`
      );
    }
    if (prior.P !== P)
      fail(
        'FINGERPRINT_MISMATCH',
        `run was started with P=${prior.P}, now P=${P}`
      );
  } else {
    await writeSidecarAtomic(runFile, { fingerprint, P, started_pairs: 0 });
  }

  const existing = new Set(await readdir(matchDir));
  const summary = { played: 0, skipped: 0, replayed: 0, quarantined: 0 };

  for (let pairIndex = 0; pairIndex < P; pairIndex++) {
    const names = [0, 1].map((g) => sidecarName(pairIndex, g));
    const present = names.map((n) => existing.has(n));

    if (present[0] && present[1]) {
      summary.skipped += 1;
      continue;
    }

    // A half-finished pair: keep the survivor as evidence, replay the whole
    // pair, and require the replay to reproduce it exactly.
    let quarantined = null;
    if (present[0] || present[1]) {
      const which = present[0] ? 0 : 1;
      const from = join(matchDir, names[which]);
      quarantined = { which, data: JSON.parse(await readFile(from, 'utf8')) };
      await rename(from, join(quarantineDir, names[which]));
      summary.quarantined += 1;
    }

    for (const gameInPair of [0, 1]) {
      const candidateIsRed = gameInPair === 0;
      const game = await playGame({
        redInference: candidateIsRed ? candidate.inference : baseline.inference,
        blackInference: candidateIsRed
          ? baseline.inference
          : candidate.inference,
        openingMoves: openings[pairIndex],
        nSimulations: nSims,
        cPuct,
      });

      const sidecar = buildSidecar({
        openingId: pairIndex, // §10: opening_id === pair_index
        openingMoves: openings[pairIndex],
        pairIndex,
        gameInPair,
        baselineModelId: baseline.modelId,
        candidateModelId: candidate.modelId,
        game,
        nSimulations: game.nSimulations,
        cPuct,
        moveTemp: game.moveTemp,
        ortVersion: ortV,
        executionCommit: commit,
      });

      if (quarantined && quarantined.which === gameInPair) {
        for (const f of REPLAY_FIELDS) {
          if (
            JSON.stringify(sidecar[f]) !== JSON.stringify(quarantined.data[f])
          ) {
            fail(
              'REPLAY_MISMATCH',
              `resumed replay of pair ${pairIndex} game ${gameInPair} differs from the quarantined ` +
                `sidecar in "${f}"; hard play is deterministic, so this means determinism does not hold`
            );
          }
        }
        summary.replayed += 1;
      }

      await writeSidecarAtomic(join(matchDir, names[gameInPair]), sidecar);
      summary.played += 1;
      if (onGameComplete)
        await onGameComplete({ pairIndex, gameInPair, sidecar });
    }
  }

  return summary;
}
