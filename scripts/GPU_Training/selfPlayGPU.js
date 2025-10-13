#!/usr/bin/env node
/**
 * GPU-accelerated self-play orchestrator
 * Spawns Swift Metal workers instead of Node.js workers
 */

import { program } from 'commander';
import { spawn } from 'child_process';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const scriptsDir = __dirname;
// Go up two levels: GPU_Training -> scripts -> Twixt_Game
const projectRoot = path.dirname(path.dirname(__dirname));

// Path to compiled Swift binary (in current directory)
const SWIFT_BINARY = path.join(__dirname, 'TwixTMetalGPU', '.build', 'release', 'twixt-metal-worker');

// ---------------- State ----------------
const workerProcs = [];
let consolidatorProc = null;
let shuttingDown = false;
let aborted = false;

// Track worker outcomes
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
  console.log(`\n[GPU orchestrator] ${signal} received — shutting down…`);

  // Stop workers
  for (const p of workerProcs) {
    try { p.kill('SIGTERM'); } catch {}
  }
  await Promise.race([
    Promise.all(workerProcs.map(p => waitCloseOrError(p))),
    new Promise(res => setTimeout(res, 3000))
  ]);

  // SIGKILL stragglers
  for (const p of workerProcs) {
    if (p.exitCode === null) { try { p.kill('SIGKILL'); } catch {} }
  }

  // Write sentinel
  if (runDir) {
    try { await fs.writeFile(path.join(runDir, 'writers.done'), ''); } catch {}
  }

  // Stop consolidator
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

  console.log('[GPU orchestrator] shutdown complete.');
  process.exit(130);
}

// ---------------- CLI ----------------
program
  .option('-g, --games <number>', 'total number of self-play games', '60')
  .option('-d, --depth <number>', 'search depth per side', '3')
  .option('-w, --workers <number>', 'number of GPU workers (default: 6, each uses GPU)', '6')
  .option('--depth-config <config>', 'multiple depths as "depth:games,depth:games" (e.g., "2:240,3:240")')
  .option('--verbose', 'print progress to stdout', false)
  .option('--build', 'build Swift binary before running', false);

program.parse(process.argv);
const opts = program.opts();

const WORKERS = parseInt(opts.workers, 10) || 6;
const VERBOSE = !!opts.verbose;
const BUILD = !!opts.build;

// Parse depth configuration
let depthConfigs = [];
if (opts.depthConfig) {
  // Parse "2:240,3:240" format
  try {
    depthConfigs = opts.depthConfig.split(',').map(pair => {
      const [depth, games] = pair.trim().split(':');
      return {
        depth: parseInt(depth, 10),
        games: parseInt(games, 10)
      };
    });
    if (depthConfigs.some(c => isNaN(c.depth) || isNaN(c.games))) {
      throw new Error('Invalid depth-config format');
    }
  } catch (err) {
    console.error('✗ Error parsing --depth-config. Expected format: "depth:games,depth:games"');
    console.error('  Example: "2:240,3:240"');
    process.exit(1);
  }
} else {
  // Single depth mode (backward compatibility)
  const TOTAL_GAMES = parseInt(opts.games, 10) || 60;
  const SEARCH_DEPTH = parseInt(opts.depth, 10) || 3;
  depthConfigs = [{ depth: SEARCH_DEPTH, games: TOTAL_GAMES }];
}

// ---------------- Main ----------------
async function main() {
  console.log('╔══════════════════════════════════════════════════════════╗');
  console.log('║   TwixT GPU-Accelerated Self-Play (Metal on M3)         ║');
  console.log('╚══════════════════════════════════════════════════════════╝\n');

  // Check if Swift binary exists
  try {
    await fs.access(SWIFT_BINARY, fs.constants.X_OK);
    console.log(`✓ Found Swift binary: ${SWIFT_BINARY}`);
  } catch {
    console.error(`✗ Swift binary not found at: ${SWIFT_BINARY}`);
    console.error('\nPlease build the Swift project first:');
    console.error('  cd scripts/GPU_Training/TwixTMetalGPU');
    console.error('  swift build -c release');
    console.error('\nOr run with --build flag to build automatically');

    if (BUILD) {
      console.log('\nBuilding Swift project...');
      await buildSwiftProject();
    } else {
      process.exit(1);
    }
  }

  const runId = Date.now().toString();
  const runDir = path.join(projectRoot, 'temp', `run-${runId}`);

  // Check for value-mode.json
  const valueModelPath = path.join(projectRoot, 'value-mode.json');
  const searchConfigPath = path.join(projectRoot, 'assets', 'js', 'ai', 'search.json');

  try {
    await fs.access(valueModelPath);
    console.log(`✓ Found value-mode.json: ${valueModelPath}`);
  } catch {
    console.log(`⚠️  value-mode.json not found at: ${valueModelPath}`);
    console.log('   GPU workers will use heuristics from search.json only\n');
  }

  // Calculate totals
  const totalGames = depthConfigs.reduce((sum, c) => sum + c.games, 0);

  console.log(`\nStarting GPU self-play:`);
  console.log(`  Total Games: ${totalGames}`);
  console.log(`  Workers:     ${WORKERS} (each using M3 GPU)`);
  if (depthConfigs.length === 1) {
    console.log(`  Depth:       ${depthConfigs[0].depth}`);
  } else {
    console.log(`  Depth Config:`);
    depthConfigs.forEach(c => {
      console.log(`    - Depth ${c.depth}: ${c.games} games`);
    });
  }
  console.log(`  Run ID:      ${runId}\n`);

  // Initialize selfplay.json
  const selfplayPath = path.join(projectRoot, 'selfplay.json');
  let gameData;
  try {
    gameData = JSON.parse(await fs.readFile(selfplayPath, 'utf8'));
    console.log(`Found existing selfplay.json with ${gameData.gameCount} games`);
  } catch {
    gameData = {
      generatedAt: new Date().toISOString(),
      gameCount: 0,
      searchDepth: depthConfigs[0].depth,
      games: []
    };
    await fs.writeFile(selfplayPath, JSON.stringify(gameData, null, 2));
    console.log('Created new selfplay.json');
  }

  // Create run directory
  await fs.mkdir(runDir, { recursive: true });

  // Setup shutdown handlers
  ['SIGINT', 'SIGTERM', 'SIGHUP'].forEach(sig => {
    process.once(sig, () => { void gracefulShutdown(sig, runDir); });
  });

  // Spawn consolidator
  console.log('Starting consolidator...');
  const consolidatorPath = path.join(path.dirname(scriptsDir), 'consolidator.js');
  consolidatorProc = spawn(process.execPath, [
    consolidatorPath,
    `--run-id=${runId}`,
    `--target-games=${totalGames}`,
    `--workers=${WORKERS * depthConfigs.length}`  // Total workers across all depths
  ], {
    stdio: VERBOSE ? 'inherit' : 'pipe',
    cwd: projectRoot
  });

  // Spawn GPU workers for each depth configuration
  let workerIdCounter = 1;

  for (let configIdx = 0; configIdx < depthConfigs.length; configIdx++) {
    const config = depthConfigs[configIdx];
    const { depth, games } = config;

    if (depthConfigs.length > 1) {
      console.log(`\n${'═'.repeat(60)}`);
      console.log(`Depth ${depth}: Spawning ${WORKERS} workers for ${games} games`);
      console.log('═'.repeat(60));
    } else {
      console.log(`\nSpawning ${WORKERS} GPU worker processes...\n`);
    }

    // Distribute games among workers
    const base = Math.floor(games / WORKERS);
    const extra = games % WORKERS;

    for (let i = 1; i <= WORKERS; i++) {
      const gamesForWorker = base + (i <= extra ? 1 : 0);
      const workerId = workerIdCounter++;

      console.log(`  GPU Worker ${workerId}: ${gamesForWorker} games @ depth ${depth}`);

      const proc = spawn(SWIFT_BINARY, [
        '-g', gamesForWorker.toString(),
        '-d', depth.toString(),
        '--core-id', workerId.toString(),
        '--run-id', runId,
        '--value-model', valueModelPath,
        '--heuristics-config', searchConfigPath,
        ...(VERBOSE ? ['--verbose'] : [])
      ], {
        stdio: VERBOSE ? 'inherit' : 'pipe',
        cwd: projectRoot
      });

      // Capture output for debugging
      if (!VERBOSE) {
        proc.stdout?.on('data', (data) => {
          console.log(`[GPU Worker ${workerId} D${depth}] ${data.toString().trim()}`);
        });
        proc.stderr?.on('data', (data) => {
          console.error(`[GPU Worker ${workerId} D${depth} ERROR] ${data.toString().trim()}`);
        });
      }

      proc.once('exit', (code, signal) => {
        workerResults.set(workerId, { code, signal });
        if (VERBOSE || code !== 0) {
          if (signal) {
            console.log(`GPU Worker ${workerId} (D${depth}) exited via signal ${signal}`);
          } else {
            console.log(`GPU Worker ${workerId} (D${depth}) exited with code ${code}`);
          }
        }
      });

      workerProcs.push(proc);
    }
  }

  console.log('\n' + '─'.repeat(60));
  console.log('GPU workers running... Press Ctrl+C to stop gracefully');
  console.log('─'.repeat(60) + '\n');

  // Wait for all workers
  console.log('Waiting for GPU workers to complete...');
  await Promise.all(workerProcs.map(p => waitCloseOrError(p)));

  if (aborted) {
    return;
  }

  // Check results
  const totalWorkers = WORKERS * depthConfigs.length;
  const results = Array.from({ length: totalWorkers }, (_, idx) =>
    workerResults.get(idx + 1) || { code: 1, signal: null }
  );
  const failedWorkers = results.filter(r => r.code !== 0);
  const allOk = failedWorkers.length === 0;

  if (allOk) {
    console.log('\n✓ All GPU workers completed successfully');
    console.log('Signaling consolidator...');
    await fs.writeFile(path.join(runDir, 'writers.done'), '');

    // Wait for consolidator
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
          console.error(`Consolidator error:`, err);
          resolve();
        });
      });
    }

    if (consolidatorExitCode === 0) {
      console.log('\n' + '═'.repeat(60));
      console.log('✅  GPU-ACCELERATED SELF-PLAY COMPLETE!');
      console.log('═'.repeat(60));
      console.log(`\nResults saved to: selfplay.json`);
      console.log(`Check GPU performance with: npm run benchmark\n`);
      process.exitCode = 0;
    } else {
      console.error('\n✗ Consolidator failed');
      process.exitCode = 1;
    }
  } else {
    console.error(`\n✗ ${failedWorkers.length}/${totalWorkers} GPU workers failed`);
    console.error('Skipping consolidator');
    process.exitCode = 1;
  }
}

async function buildSwiftProject() {
  const swiftDir = path.join(__dirname, 'TwixTMetalGPU');
  console.log(`Building in ${swiftDir}...`);

  return new Promise((resolve, reject) => {
    const proc = spawn('swift', ['build', '-c', 'release'], {
      cwd: swiftDir,
      stdio: 'inherit'
    });

    proc.on('close', (code) => {
      if (code === 0) {
        console.log('✓ Swift build successful\n');
        resolve();
      } else {
        reject(new Error(`Swift build failed with code ${code}`));
      }
    });

    proc.on('error', reject);
  });
}

main().catch(err => {
  console.error('\n✗ Error in GPU self-play:', err);
  process.exit(1);
});
