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
  expectBaselineId = BASELINE_MODEL_ID,
  expectCandidateId = CANDIDATE_MODEL_ID,
  onGameComplete = null,
}) {
  const matchDir = join(runDir, 'match');
  const quarantineDir = join(runDir, 'quarantine');
  await mkdir(matchDir, { recursive: true });
  await mkdir(quarantineDir, { recursive: true });

  const baseline = await loadModel(baselineDir);
  const candidate = await loadModel(candidateDir);
  // Bind identity to ROLE before a single game is played. Both directories can
  // be internally valid and still be the wrong two models, or swapped; without
  // this the error surfaces only at analysis, after the match time is spent.
  if (baseline.modelId !== expectBaselineId)
    fail(
      'WRONG_BASELINE_MODEL',
      `baseline dir holds ${baseline.modelId}, expected ${expectBaselineId}`
    );
  if (candidate.modelId !== expectCandidateId)
    fail(
      'WRONG_CANDIDATE_MODEL',
      `candidate dir holds ${candidate.modelId}, expected ${expectCandidateId}`
    );
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

  const readJson = async (path) => JSON.parse(await readFile(path, 'utf8'));

  /**
   * Every quarantined observation of one (pair, slot).
   *
   * Names are unique and never reused, so a second interruption cannot
   * overwrite the first record, and a crash between quarantining and replaying
   * still leaves the owed comparison discoverable on the next restart.
   */
  const quarantineRecords = async (pairIndex, gameInPair) => {
    const prefix = `${sidecarName(pairIndex, gameInPair).replace(/\.json$/, '')}.q`;
    const files = (await readdir(quarantineDir)).filter(
      (n) => n.startsWith(prefix) && n.endsWith('.json')
    );
    files.sort();
    const out = [];
    for (const f of files)
      out.push({ file: f, data: await readJson(join(quarantineDir, f)) });
    return out;
  };

  const nextQuarantinePath = async (pairIndex, gameInPair, tag) => {
    const base = sidecarName(pairIndex, gameInPair).replace(/\.json$/, '');
    const taken = new Set(await readdir(quarantineDir));
    for (let n = 0; n < 10000; n++) {
      const name = `${base}.q${String(n).padStart(2, '0')}${tag}.json`;
      if (!taken.has(name)) return join(quarantineDir, name);
    }
    fail(
      'QUARANTINE_FULL',
      `cannot allocate a quarantine name for pair ${pairIndex}`
    );
  };

  /**
   * Is this existing sidecar evidence from THIS run, at THIS position?
   *
   * Checked before a pair is skipped OR quarantined. Trusting a filename would
   * let a completed pair carrying a different `execution_commit` be skipped
   * silently, which is precisely the mixed-implementation run the fingerprint
   * exists to prevent.
   */
  const validateExisting = (sidecar, pairIndex, gameInPair, where) => {
    const bad = (why, extra = {}) =>
      fail(
        'EXISTING_SIDECAR_INVALID',
        `${where}: ${why} — ${JSON.stringify({ pairIndex, gameInPair, ...extra })}`
      );
    if (!sidecar || typeof sidecar !== 'object') bad('not an object');
    if (sidecar.kind !== 'match')
      bad('kind is not "match"', { kind: sidecar.kind });
    if (sidecar.schema !== SCHEMA)
      bad('wrong schema', { schema: sidecar.schema });
    if (sidecar.pair_index !== pairIndex)
      bad('pair_index does not match its filename');
    if (sidecar.game_in_pair !== gameInPair)
      bad('game_in_pair does not match its filename');
    if (sidecar.opening_id !== pairIndex) bad('opening_id is not pair_index');
    const expectedHash = sha256(JSON.stringify(openings[pairIndex]));
    if (sidecar.opening_sha256 !== expectedHash)
      bad('opening hash does not match the pool');
    for (const f of FINGERPRINT_FIELDS) {
      if (sidecar[f] !== fingerprint[f]) {
        bad('fingerprint field differs from this run', {
          field: f,
          found: sidecar[f],
          expected: fingerprint[f],
        });
      }
    }
    const expectRed =
      gameInPair === 0
        ? fingerprint.candidate_model_id
        : fingerprint.baseline_model_id;
    const expectBlack =
      gameInPair === 0
        ? fingerprint.baseline_model_id
        : fingerprint.candidate_model_id;
    if (
      sidecar.red_model_id !== expectRed ||
      sidecar.black_model_id !== expectBlack
    ) {
      bad('colour assignment contradicts game_in_pair', {
        red: sidecar.red_model_id,
        black: sidecar.black_model_id,
      });
    }
  };

  const differsFrom = (a, b) =>
    REPLAY_FIELDS.find((f) => JSON.stringify(a[f]) !== JSON.stringify(b[f]));

  const summary = {
    played: 0,
    skipped: 0,
    replayed: 0,
    quarantined: 0,
    verifiedFromDisk: 0,
  };

  for (let pairIndex = 0; pairIndex < P; pairIndex++) {
    const names = [0, 1].map((g) => sidecarName(pairIndex, g));
    const onDisk = new Set(await readdir(matchDir));
    const present = names.map((n) => onDisk.has(n));

    const priorRecords = [
      await quarantineRecords(pairIndex, 0),
      await quarantineRecords(pairIndex, 1),
    ];

    if (present[0] && present[1]) {
      // Validate before skipping — existence of a filename is not evidence.
      for (const g of [0, 1]) {
        const sidecar = await readJson(join(matchDir, names[g]));
        validateExisting(sidecar, pairIndex, g, names[g]);
        // A completed pair that also has quarantined history must still agree
        // with it. Re-checking from disk costs nothing and keeps the owed
        // comparison honoured across any number of restarts.
        for (const rec of priorRecords[g]) {
          const field = differsFrom(sidecar, rec.data);
          if (field) {
            fail(
              'REPLAY_MISMATCH',
              `completed pair ${pairIndex} game ${g} differs from quarantined ${rec.file} in "${field}"`
            );
          }
          summary.verifiedFromDisk += 1;
        }
      }
      summary.skipped += 1;
      continue;
    }

    // Incomplete pair. Quarantine any survivor under a fresh name, then replay
    // the whole pair: a pair is the statistical unit and is never half-counted.
    for (const g of [0, 1]) {
      if (!present[g]) continue;
      const from = join(matchDir, names[g]);
      const sidecar = await readJson(from);
      validateExisting(sidecar, pairIndex, g, names[g]);
      await rename(from, await nextQuarantinePath(pairIndex, g, ''));
      summary.quarantined += 1;
    }
    // Re-read after quarantining, so the survivor just moved is included.
    const owed = [
      await quarantineRecords(pairIndex, 0),
      await quarantineRecords(pairIndex, 1),
    ];

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

      // The replay must reproduce EVERY prior observation of this slot. If two
      // quarantined copies already disagree with each other, that is itself a
      // determinism failure and this catches it too.
      for (const rec of owed[gameInPair]) {
        const field = differsFrom(sidecar, rec.data);
        if (field) {
          // Persist the divergent replay BEFORE aborting. Both copies must
          // survive: the disagreement is the evidence, and hard play being
          // deterministic makes it the most serious defect this run can find.
          const divergentPath = await nextQuarantinePath(
            pairIndex,
            gameInPair,
            '.divergent'
          );
          await writeSidecarAtomic(divergentPath, sidecar);
          fail(
            'REPLAY_MISMATCH',
            `resumed replay of pair ${pairIndex} game ${gameInPair} differs from quarantined ` +
              `${rec.file} in "${field}"; hard play is deterministic, so this means determinism ` +
              `does not hold. Divergent replay retained at ${divergentPath}`
          );
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
