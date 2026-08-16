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
  P_DECISION_PATH,
  P_DECISION_RELPATH,
  REPO_ROOT,
  TIMING_GAMES,
  TIMING_OPENING_MAPPING,
  buildDecision,
  sha256,
} from './p_decision.mjs';

const POOL_RELPATH = 'tests/product_match/openings.json';

/** The ten timing games, in fixed order, derived from the frozen mapping. */
export function timingSchedule() {
  const schedule = [];
  for (const openingId of TIMING_OPENING_MAPPING.baseline_self_play) {
    schedule.push({ openingId, arm: 'baseline_self_play' });
  }
  for (const openingId of TIMING_OPENING_MAPPING.candidate_self_play) {
    schedule.push({ openingId, arm: 'candidate_self_play' });
  }
  if (schedule.length !== TIMING_GAMES) {
    throw new Error(
      `timing schedule has ${schedule.length} games, expected ${TIMING_GAMES}`
    );
  }
  return schedule;
}

export const timingSidecarName = (index, openingId) =>
  `timing_${String(index).padStart(2, '0')}_opening_${openingId}.json`;

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
  now = () => Date.now(),
  playGameFn = playGame,
}) {
  const openingMoves = openingMovesFrom(openings);
  const timingDir = join(runDir, 'timing');
  await mkdir(timingDir, { recursive: true });

  const schedule = timingSchedule();
  const evidence = [];

  // The clock starts here: both sessions are already loaded and both contracts
  // already asserted by the caller, so one-off model-load cost is excluded
  // while still having been paid.
  const t0 = now();

  for (let i = 0; i < schedule.length; i++) {
    const { openingId, arm } = schedule[i];
    const model = arm === 'baseline_self_play' ? baseline : candidate;

    const game = await playGameFn({
      redInference: model.inference,
      blackInference: model.inference,
      openingMoves: openingMoves[openingId],
      nSimulations,
      cPuct,
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
    // Hash what is on disk, so the evidence digest describes the committed
    // bytes rather than an in-memory object that may not match them.
    evidence.push({
      file,
      sha256: sha256(await readFile(join(timingDir, file))),
    });
  }

  // …and stops only after the last sidecar's atomic rename has completed. The
  // match pays search and write, so both are inside the span.
  const totalSequentialWallMs = now() - t0;

  const decision = buildDecision({
    totalSequentialWallMs,
    timingEvidence: evidence,
    openingPoolSha256: sha256(await readFile(join(REPO_ROOT, POOL_RELPATH))),
    executionCommit: commit,
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

  const openings = JSON.parse(
    await readFile(join(REPO_ROOT, POOL_RELPATH), 'utf8')
  );
  const policy = hardPolicy();
  const commit = executionCommit(); // refuses a dirty worktree
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
  });

  await writeFile(P_DECISION_PATH, `${JSON.stringify(decision, null, 2)}\n`);
  console.log(
    `\nthroughput ${decision.measured.games_per_hour.toFixed(2)} games/hour`
  );
  console.log(`selected P = ${decision.selected_p}`);
  console.log(
    `\nwrote ${P_DECISION_RELPATH} — COMMIT IT before any match game.`
  );
}

const isMain =
  process.argv[1] && import.meta.url === `file://${process.argv[1]}`;
if (isMain) await main();
