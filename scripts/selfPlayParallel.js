import { program } from 'commander';
import { spawn } from 'child_process';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const scriptsDir = __dirname;            // This file is in scripts/ directory
const projectRoot = path.dirname(__dirname); // Parent directory (project root)

// ---------------- State ----------------
const workerProcs = [];
let consolidatorProc = null;
let shuttingDown = false;   // controls one-time shutdown sequence
let aborted = false;        // indicates we received a terminating signal

// Track worker outcomes (index -> { code, signal })
const workerResults = new Map();

// ---------------- Helpers ----------------
function waitCloseOrError(child) {
  return new Promise(res => {
    child.once('close', res);
    child.once('error', res);
  });
}

async function gracefulShutdown(signal = 'SIGINT', runDir = null) {
  if (shuttingDown) return;
  shuttingDown = true;
  aborted = true;
  console.log(`\n[orchestrator] ${signal} received — shutting down…`);

  // 1) Stop workers first so temp files stop growing
  for (const p of workerProcs) {
    try { p.kill('SIGTERM'); } catch {}
  }
  await Promise.race([
    Promise.all(workerProcs.map(p => waitCloseOrError(p))),
    new Promise(res => setTimeout(res, 3000))
  ]);
  // SIGKILL any stragglers
  for (const p of workerProcs) {
    if (p.exitCode === null) { try { p.kill('SIGKILL'); } catch {} }
  }

  // 2) Write sentinel so consolidator knows no more writers are coming
  if (runDir) {
    try { await fs.writeFile(path.join(runDir, 'writers.done'), ''); } catch {}
  }

  // 3) Ask consolidator to finalize and exit
  if (consolidatorProc) {
    try { consolidatorProc.kill('SIGINT'); } catch {}
    await Promise.race([
      waitCloseOrError(consolidatorProc),
      new Promise(res => setTimeout(res, 5000))
    ]);
    if (consolidatorProc.exitCode === null) {
      try { consolidatorProc.kill('SIGKILL'); } catch {}
    }
  }

  console.log('[orchestrator] shutdown complete.');
  process.exit(130); // 128 + SIGINT
}

// ---------------- CLI ----------------
program
  .option('-g, --games <number>', 'total number of self-play games', '60')
  .option('-d, --depth <number>', 'search depth per side', '3')
  .option('-w, --workers <number>', 'number of worker processes', '12')
  .option('--depth-config <config>',
    'comma-separated depth plan, e.g. "2:100,3:80" (runs batches sequentially)')
  .option('--verbose', 'print progress to stdout', false);

program.parse(process.argv);
const opts = program.opts();

const TOTAL_GAMES_RAW = parseInt(opts.games, 10) || 60;
const TOTAL_GAMES  = TOTAL_GAMES_RAW;
if (TOTAL_GAMES % 2 !== 0) {
  throw new Error('Total games must be even to balance starting colours.');
}
const SEARCH_DEPTH = parseInt(opts.depth, 10) || 3;
const VERBOSE      = !!opts.verbose;
const WORKERS      = Math.max(1, parseInt(opts.workers, 10) || 12);

function parseDepthConfig(raw) {
  if (!raw) return null;
  const batches = [];
  for (const segment of raw.split(',')) {
    const trimmed = segment.trim();
    if (!trimmed) continue;
    const [depthStr, gamesStr] = trimmed.split(':');
    const depth = parseInt(depthStr, 10);
    const games = parseInt(gamesStr, 10);
    if (!Number.isFinite(depth) || depth <= 0 || !Number.isFinite(games) || games <= 0) {
      throw new Error(`Invalid depth-config entry "${segment}". Expected format depth:games with positive integers.`);
    }
    if (games % 2 !== 0) {
      throw new Error(`Depth batch "${segment}" must have an even number of games to balance starting colours.`);
    }
    batches.push({ depth, games });
  }
  if (batches.length === 0) {
    throw new Error('depth-config did not contain any valid depth:games pairs.');
  }
  return batches;
}

function buildBalancedStartPlan(count) {
  if (count % 2 !== 0) {
    throw new Error('Balanced start plan requires an even game count.');
  }
  const half = count / 2;
  const plan = [];
  for (let i = 0; i < half; i++) plan.push('red');
  for (let i = 0; i < half; i++) plan.push('black');
  for (let i = plan.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [plan[i], plan[j]] = [plan[j], plan[i]];
  }
  return plan;
}

// ---------------- Main ----------------
async function runBatch(totalGames, depth, runLabel) {
  workerProcs.length = 0;
  workerResults.clear();
  consolidatorProc = null;
  shuttingDown = false;
  aborted = false;
  const activeWorkerIds = [];

  const runId  = Date.now().toString();
  const runDir = path.join('temp', `run-${runId}`);

  console.log(`\n=== Starting batch${runLabel ? ` ${runLabel}` : ''}: depth ${depth}, ${totalGames} games across ${WORKERS} workers ===`);
  console.log(`Run ID: ${runId}`);

  const startPlan = buildBalancedStartPlan(totalGames);

  // Initialize/read selfplay.json
  let gameData;
  try {
    gameData = JSON.parse(await fs.readFile('selfplay.json', 'utf8'));
    console.log(`Found existing selfplay.json with ${gameData.gameCount} games`);
  } catch {
    gameData = {
      generatedAt: new Date().toISOString(),
      gameCount: 0,
      searchDepth: SEARCH_DEPTH,
      games: []
    };
    await fs.writeFile('selfplay.json', JSON.stringify(gameData, null, 2));
    console.log('Created new selfplay.json');
  }

  // Precise work distribution (e.g., 61 games → 5 get 10, 1 gets 11)
  const base  = Math.floor(totalGames / WORKERS);
  const extra = totalGames % WORKERS;

  // Create run directory
  await fs.mkdir(runDir, { recursive: true });

  // Ensure Ctrl+C shuts everything down cleanly
  ['SIGINT', 'SIGTERM', 'SIGHUP'].forEach(sig => {
    process.once(sig, () => { void gracefulShutdown(sig, runDir); });
  });

  // Spawn consolidator first
  console.log('Starting consolidator...');
  consolidatorProc = spawn(process.execPath, [
    path.join(scriptsDir, 'consolidator.js'),
    `--run-id=${runId}`,
    `--target-games=${totalGames}`,
    `--workers=${WORKERS}`
  ], {
    stdio: VERBOSE ? 'inherit' : 'pipe',
    cwd: projectRoot
  });

  // Spawn worker processes
  console.log(`Spawning up to ${WORKERS} worker processes...`);
  for (let i = 1; i <= WORKERS; i++) {
    const gamesForWorker = base + (i <= extra ? 1 : 0);
    if (gamesForWorker === 0) {
      if (VERBOSE) {
        console.log(`  Worker ${i}: skipped (no games assigned in this batch)`);
      }
      continue;
    }
    console.log(`  Worker ${i}: ${gamesForWorker} games`);

    const workerPlan = startPlan.splice(0, gamesForWorker);

    const proc = spawn(process.execPath, [
      path.join(scriptsDir, 'selfPlay.js'),
      '-g', gamesForWorker.toString(),
      '-d', depth.toString(),
      '--verbose',
      `--core-id=${i}`,
      `--run-id=${runId}`
    ], {
      stdio: VERBOSE ? 'inherit' : 'pipe',
      cwd: projectRoot,
      env: {
        ...process.env,
        START_PLAN: workerPlan.join(',')
      }
    });

    // Record exit details (code & signal)
    proc.once('exit', (code, signal) => {
      workerResults.set(i, { code, signal });
      if (VERBOSE) {
        if (signal) {
          console.log(`Worker ${i} exited via signal ${signal}`);
        } else {
          console.log(`Worker ${i} exited with code ${code}`);
        }
      }
    });

    workerProcs.push(proc);
    activeWorkerIds.push(i);
  }

  if (workerProcs.length === 0) {
    console.log('No workers were launched for this batch (no games assigned). Skipping to next batch.');
    return;
  }

  // Wait for ALL active workers to exit (success or error)
  console.log('Waiting for all workers to complete...');
  await Promise.all(workerProcs.map(p => waitCloseOrError(p)));

  // If we were aborted, gracefulShutdown already handled sentinel & consolidator
  if (aborted) {
    // Exit path already managed by gracefulShutdown (process.exit(130))
    return;
  }

  // Assess worker outcomes (only 0 = success)
  const results = activeWorkerIds.map(id => workerResults.get(id) || { code: 1, signal: null });
  const failedWorkers = results.filter(r => r.code !== 0);
  const allOk = failedWorkers.length === 0;

  if (allOk) {
    console.log('All workers completed. Signaling consolidator...');
    await fs.writeFile(path.join(runDir, 'writers.done'), '');

    // Wait for consolidator to finish
    let consolidatorExitCode = 0;
    if (consolidatorProc.exitCode !== null) {
      consolidatorExitCode = consolidatorProc.exitCode;
      console.log(`Consolidator already exited with code ${consolidatorExitCode}`);
    } else {
      await new Promise(resolve => {
        consolidatorProc.once('close', (code) => {
          consolidatorExitCode = code ?? 1;
          console.log(`Consolidator exited with code ${code}`);
          resolve();
        });
        consolidatorProc.once('error', (err) => {
          consolidatorExitCode = 1;
          console.error(`Consolidator process error:`, err);
          resolve();
        });
      });
    }

    if (consolidatorExitCode === 0) {
      console.log('✅ Parallel self-play complete! Check selfplay.json for results.');
    } else {
      console.error('Consolidator failed.');
      throw new Error('Consolidator failed');
    }
  } else {
    console.error(`Process failures detected: ${failedWorkers.length}/${WORKERS} workers failed.`);
    console.error('Skipping consolidator.');
    throw new Error('One or more workers failed');
  }
}

async function main() {
  try {
    const depthPlan = parseDepthConfig(opts.depthConfig);
    if (depthPlan && depthPlan.length > 0) {
      for (let i = 0; i < depthPlan.length; i++) {
        const batch = depthPlan[i];
        await runBatch(batch.games, batch.depth, `${i + 1}/${depthPlan.length}`);
      }
      console.log('\nAll depth batches completed successfully.');
    } else {
      await runBatch(TOTAL_GAMES, SEARCH_DEPTH);
    }
    process.exitCode = 0;
  } catch (err) {
    console.error('Error in parallel self-play:', err.message ?? err);
    process.exitCode = 1;
  }
}

main();
if (TOTAL_GAMES % 2 !== 0) {
  throw new Error('Total games must be even to balance starting colours.');
}
