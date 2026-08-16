#!/usr/bin/env node
/**
 * Timing smoke: measure product-stack throughput, then derive `P` from it.
 *
 *   node tests/product_match/timing.mjs <run_dir>
 *
 * The smoke plays ten self-play games on the reserved openings `200…209` — five
 * baseline-versus-baseline, five candidate-versus-candidate. Self-play on both
 * sides is what makes it OUTCOME-BLIND: a model playing itself yields no
 * comparative information, so `P` cannot be chosen with any knowledge of the
 * matchup.
 *
 * It shares `playGame`, model loading and the atomic sidecar writer with the
 * match harness on purpose. A timing measurement of a different code path would
 * measure the wrong thing.
 *
 * RUNNING THIS IS SEPARATELY AUTHORIZED. The capability exists; invoking it on
 * the reserved openings is a distinct gate. Nothing here is called by any test:
 * `runTimingSmoke` takes injected clock and game seams so its logic can be
 * exercised without playing a game, and the CLI below is the only production
 * entry point.
 *
 * Specification: docs/superpowers/2026-08-14-product-stack-comparison-specification.md §7.3
 */
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';

import {
  BASELINE_MODEL_ID,
  CANDIDATE_MODEL_ID,
  SCHEMA as MATCH_SCHEMA,
  executionCommit,
  hardPolicy,
  loadModel,
  ortVersion,
  playGame,
  writeSidecarAtomic,
} from './harness.mjs';
import { openingMovesFrom } from './generate_openings.mjs';
import {
  EXECUTION_SURFACE_FILES,
  P_DECISION_PATH,
  P_DECISION_RELPATH,
  POOL_RELPATH,
  REPO_ROOT,
  TIMING_GAMES,
  buildDecision,
  executionSurfaceDigest,
  readCommittedBlob,
  sha256,
  timingSchedule,
  timingSidecarName,
} from './p_decision.mjs';

export { timingSchedule, timingSidecarName };

/** Milliseconds from a MONOTONIC source, immune to wall-clock adjustment. */
export const monotonicMs = () => Number(process.hrtime.bigint() / 1000n) / 1000;

/**
 * Run the smoke and produce a decision.
 *
 * Execution is one process, sequential, no concurrency, at the product's own
 * ORT configuration — otherwise the measured throughput would describe a
 * configuration the match will not use.
 *
 * `now` and `playGameFn` are injected so the timing arithmetic, schedule,
 * sidecar shape and decision derivation can be tested without playing a game.
 * Production passes neither.
 */
export async function runTimingSmoke({
  runDir,
  openings,
  baseline,
  candidate,
  nSimulations,
  cPuct = 1.5,
  moveTemp,
  ortVersion: ortV,
  executionCommit: commit,
  poolSha256,
  executionSurfaceSha256,
  now = monotonicMs,
  playGameFn = playGame,
}) {
  // The pool is normalized ONCE, before any game, and the digest describes the
  // exact bytes these games are played from. Hashing the file afterwards could
  // describe something edited during a multi-hour run.
  const openingMoves = openingMovesFrom(openings);
  const timingDir = join(runDir, 'timing');
  await mkdir(timingDir, { recursive: true });

  const schedule = timingSchedule();
  const written = [];

  let t0 = null;
  let t1 = null;

  for (let i = 0; i < schedule.length; i++) {
    const { openingId, arm } = schedule[i];
    const model = arm === 'baseline_self_play' ? baseline : candidate;

    const game = await playGameFn({
      redInference: model.inference,
      blackInference: model.inference,
      openingMoves: openingMoves[openingId],
      nSimulations,
      cPuct,
      // The clock starts at the exact preregistered seam — immediately before
      // game 1's first search — not before the MCTS objects are built or the
      // opening replayed. Both would time work the match does not repeat.
      onFirstSearch:
        i === 0
          ? () => {
              t0 = now();
            }
          : null,
    });

    const sidecar = {
      kind: 'timing',
      schema: MATCH_SCHEMA,
      arm,
      timing_index: i,
      opening_id: openingId,
      opening_sha256: sha256(JSON.stringify(openingMoves[openingId])),
      red_model_id: model.modelId,
      black_model_id: model.modelId,
      moves: game.moves,
      result: game.result,
      termination: game.termination,
      ply_count: game.plyCount,
      n_simulations: game.nSimulations,
      c_puct: cPuct,
      move_temp: game.moveTemp,
      ort_version: ortV,
      ort_config: 'no options supplied',
      execution_commit: commit,
      elapsed_ms: game.elapsedMs,
      note: 'Self-play. Carries no comparative information and is never read by the match analyser.',
    };

    const file = timingSidecarName(i, openingId);
    await writeSidecarAtomic(join(timingDir, file), sidecar);
    written.push(file);

    // …and stops the instant the tenth rename completes. Digesting the evidence
    // is bookkeeping the match does not perform, so it happens after.
    if (i === schedule.length - 1) t1 = now();
  }

  if (t0 === null || t1 === null) {
    throw new Error(
      'timing span was never opened or closed; the pre-search seam did not fire'
    );
  }
  const totalSequentialWallMs = t1 - t0;

  const evidence = [];
  for (const file of written) {
    evidence.push({
      file,
      sha256: sha256(await readFile(join(timingDir, file))),
    });
  }

  const decision = buildDecision({
    totalSequentialWallMs,
    timingEvidence: evidence,
    openingPoolSha256: poolSha256,
    executionCommit: commit,
    executionSurfaceSha256,
    baselineModelId: baseline.modelId,
    candidateModelId: candidate.modelId,
    ortVersion: ortV,
    ortConfig: 'no options supplied',
    nSimulations,
    cPuct,
    moveTemp,
  });

  return { decision, totalSequentialWallMs, evidence, schedule };
}

// --- production entry point --------------------------------------------------
// Loads the real models and plays the ten reserved-opening games. Invoking this
// is a separate authorization from writing it.

async function main() {
  const runDir = process.argv[2];
  if (!runDir) {
    console.error('usage: timing.mjs <run_dir>');
    process.exit(2);
  }

  // Everything the decision binds is read from the COMMIT, before any game, so
  // a long run cannot silently span a repository change.
  const commit = executionCommit(); // refuses a dirty worktree
  const poolBytes = readCommittedBlob(POOL_RELPATH, commit, REPO_ROOT);
  const poolSha256 = sha256(poolBytes);
  const executionSurfaceSha256 = executionSurfaceDigest(commit, REPO_ROOT);
  const openings = JSON.parse(poolBytes.toString('utf8'));

  const policy = hardPolicy();
  const ortV = await ortVersion();

  // Both models loaded and both contracts asserted BEFORE the clock starts.
  const baseline = await loadModel(
    join(REPO_ROOT, 'models', BASELINE_MODEL_ID)
  );
  const candidate = await loadModel(
    join(REPO_ROOT, 'models', CANDIDATE_MODEL_ID)
  );
  if (
    baseline.modelId !== BASELINE_MODEL_ID ||
    candidate.modelId !== CANDIDATE_MODEL_ID
  ) {
    console.error('model directories do not hold the expected roles');
    process.exit(1);
  }

  const { decision } = await runTimingSmoke({
    runDir,
    openings,
    baseline,
    candidate,
    nSimulations: policy.nSims,
    moveTemp: policy.moveTemp,
    ortVersion: ortV,
    executionCommit: commit,
    poolSha256,
    executionSurfaceSha256,
  });

  // The repository must not have moved under a multi-hour run: a decision
  // describing one commit while the tree sits at another is not a decision
  // about this code.
  const endCommit = executionCommit();
  if (endCommit !== commit) {
    console.error(
      `\nREFUSED: repository moved during the run (${commit} -> ${endCommit})`
    );
    process.exit(1);
  }
  if (executionSurfaceDigest(endCommit, REPO_ROOT) !== executionSurfaceSha256) {
    console.error('\nREFUSED: the execution surface changed during the run');
    process.exit(1);
  }

  await writeFile(P_DECISION_PATH, `${JSON.stringify(decision, null, 2)}\n`);
  console.log(
    `\nthroughput ${decision.measured.games_per_hour.toFixed(2)} games/hour`
  );
  console.log(`selected P = ${decision.selected_p}`);
  console.log(
    `surface    ${executionSurfaceSha256.slice(0, 16)} over ${EXECUTION_SURFACE_FILES.length} files`
  );
  console.log(
    `\nwrote ${P_DECISION_RELPATH} — COMMIT IT before any match game.`
  );
}

const isMain =
  process.argv[1] && import.meta.url === `file://${process.argv[1]}`;
if (isMain) await main();
