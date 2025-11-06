import { program } from 'commander';
import path from 'path';
import fs from 'fs/promises';
import { createReadStream } from 'fs';

// Fixed arg parsing with commander
program
  .option('--run-id <string>', 'run identifier')
  .option('--target-games <number>', 'expected total games', '0')
  .option('--workers <number>', 'number of worker files to watch', '6');

program.parse();
const { runId, targetGames, workers: workersOpt } = program.opts();
const workers = Math.max(1, parseInt(workersOpt || '6', 10) || 6);

if (!runId) {
  console.error('Error: --run-id is required');
  process.exit(1);
}

const runDir = path.join('temp', `run-${runId}`);

console.log(
  `Consolidator starting for run ${runId}, target: ${targetGames} games`
);

// --- Helpers ---------------------------------------------------------------

// Make each move "lean" by dropping heavy fields and ensuring counts exist
function toLeanMoves(moves) {
  if (!Array.isArray(moves)) return [];
  return moves.map((m) => {
    const player = m.player;
    const opp = player === 'red' ? 'black' : 'red';

    // start with provided context (copy), or an empty one
    const ctx = m.featureContext ? { ...m.featureContext } : {};

    // Ensure peg counts exist; if missing and a board snapshot exists, compute them
    if (
      (ctx.playerPegCount == null || ctx.opponentPegCount == null) &&
      Array.isArray(m.board)
    ) {
      let playerCount = 0;
      let oppCount = 0;
      for (const row of m.board) {
        for (const cell of row) {
          if (cell === player) playerCount++;
          else if (cell === opp) oppCount++;
        }
      }
      if (ctx.playerPegCount == null) ctx.playerPegCount = playerCount;
      if (ctx.opponentPegCount == null) ctx.opponentPegCount = oppCount;
    }

    return {
      turn: m.turn,
      player: m.player,
      move: m.move,
      heuristics: m.heuristics ?? m.features ?? null,
      featureContext: ctx,
      valueModel: m.valueModel ?? null,
      heuristicScore: m.heuristicScore ?? null,
      // deliberately drop: board, lastMoveTrace, any other bulky fields
    };
  });
}

// Read only new complete lines from a JSONL file since the last offset
async function readNew(jsonlPath, offsetPath) {
  let lastOffset = 0;
  try {
    lastOffset = parseInt(await fs.readFile(offsetPath, 'utf8'));
  } catch {}

  // Handle missing temp files gracefully
  const stats = await fs.stat(jsonlPath).catch(() => ({ size: 0 }));
  if (stats.size <= lastOffset) return { lines: [], upto: lastOffset };

  const stream = createReadStream(jsonlPath, {
    start: lastOffset,
    end: stats.size - 1,
    encoding: 'utf8',
  });

  let buffer = '';
  const lines = [];

  for await (const chunk of stream) {
    buffer += chunk;
    const parts = buffer.split('\n');
    buffer = parts.pop(); // keep partial line
    for (const line of parts) {
      if (line.trim()) lines.push(line);
    }
  }

  const upto = stats.size - buffer.length; // byte position after last complete newline
  return { lines, upto };
}

async function checkAllOffsetsAtEOF(workers, runDir) {
  for (let coreId = 1; coreId <= workers; coreId++) {
    const jsonlPath = path.join(runDir, `temp-core-${coreId}.jsonl`);
    const offsetPath = path.join(runDir, `temp-core-${coreId}.offset`);
    const [st, off] = await Promise.all([
      fs.stat(jsonlPath).catch(() => ({ size: 0 })),
      fs.readFile(offsetPath, 'utf8').catch(() => '0'),
    ]);
    if (st.size > parseInt(off, 10)) return false;
  }
  return true;
}

// --------------------------------------------------------------------------

async function main() {
  // Load selfplay.json once into memory
  let mainData;
  try {
    mainData = JSON.parse(await fs.readFile('selfplay.json', 'utf8'));
  } catch {
    mainData = {
      generatedAt: new Date().toISOString(),
      gameCount: 0,
      searchDepth: 3,
      games: [],
    };
  }

  let nextGameNumber = mainData.gameCount + 1;
  const initialCount = mainData.gameCount;

  console.log(`Starting from game number ${nextGameNumber}`);

  // Seed deduplication from existing games (critical for restarts)
  const processedGames = new Set();
  for (const game of mainData.games) {
    if (game.meta) {
      processedGames.add(
        `${game.meta.runId}-${game.meta.coreId}-${game.meta.seq}`
      );
    }
  }

  const toCommit = [];
  const pendingUpto = {}; // Track byte positions for two-phase commit
  let lastCommitAt = 0; // wall-clock of last commit

  // Atomic batch commit with safety; compact output (no pretty-print)
  async function maybeCommit(force = false) {
    if (toCommit.length === 0) return;
    // lower threshold to reduce memory spikes
    if (!force && toCommit.length < 2) return;

    const batch = toCommit.splice(0);

    // Update in-memory data
    mainData.games.push(...batch);
    mainData.gameCount = mainData.games.length;
    mainData.generatedAt = new Date().toISOString();

    // Atomic write: tmp → fdatasync → rename; COMPACT JSON (no pretty-print)
    const tmpPath = 'selfplay.tmp.json';
    const fh = await fs.open(tmpPath, 'w');
    await fh.write(JSON.stringify(mainData));
    await fh.sync();
    await fh.close();

    await fs.rename(tmpPath, 'selfplay.json');

    // Directory fsync for full POSIX safety
    const dirFh = await fs.open('.', 'r');
    await dirFh.sync();
    await dirFh.close();

    // Snapshot BEFORE clearing pendingUpto
    const snapshotOffsets = { ...pendingUpto };

    // Two-phase: only advance offsets AFTER successful commit
    for (const [coreId, upto] of Object.entries(pendingUpto)) {
      const offsetPath = path.join(runDir, `temp-core-${coreId}.offset`);
      await fs.writeFile(offsetPath, String(upto));
    }
    Object.keys(pendingUpto).forEach((k) => delete pendingUpto[k]);

    // Update run.meta.json (small file; pretty ok)
    const metaPath = path.join(runDir, 'run.meta.json');
    await fs.writeFile(
      metaPath,
      JSON.stringify(
        {
          ingested: mainData.gameCount - initialCount,
          nextGameNumber,
          lastCommitAt: new Date().toISOString(),
          perCoreOffsets: snapshotOffsets,
        },
        null,
        2
      )
    );

    lastCommitAt = Date.now();
    console.log(
      `Committed ${batch.length} games, total: ${mainData.gameCount}`
    );
  }

  // Polling loop
  const pollInterval = setInterval(async () => {
    try {
      // Read from all temp files
      for (let coreId = 1; coreId <= workers; coreId++) {
        const jsonlPath = path.join(runDir, `temp-core-${coreId}.jsonl`);
        const offsetPath = path.join(runDir, `temp-core-${coreId}.offset`);

        const { lines, upto } = await readNew(jsonlPath, offsetPath);
        if (lines.length > 0) {
          pendingUpto[coreId] = upto; // Store for two-phase commit
        }

        for (const line of lines) {
          try {
            const game = JSON.parse(line);

            // Use provenance for deduplication
            const gameKey = `${game.meta?.runId}-${game.meta?.coreId}-${game.meta?.seq}`;
            if (processedGames.has(gameKey)) continue;

            // Normalize to lean moves (defensive — in case workers didn't prune)
            if (Array.isArray(game.moves)) {
              // Only transform if heavy fields likely present
              const hasHeavy = game.moves.some(
                (m) => m && (m.board || m.lastMoveTrace)
              );
              game.moves = hasHeavy ? toLeanMoves(game.moves) : game.moves;
            }

            // Assign sequential game number and buffer
            game.gameNumber = nextGameNumber++;
            processedGames.add(gameKey);
            toCommit.push(game);
          } catch (err) {
            // Quarantine corrupt line
            const badPath = path.join(runDir, `temp-core-${coreId}.bad`);
            await fs.appendFile(badPath, line + '\n');
            console.warn(`Corrupt line quarantined: ${err.message}`);
          }
        }
      }

      // Batch commit logic: commit frequently to keep memory steady
      const writersFinished = await fs
        .access(path.join(runDir, 'writers.done'))
        .then(
          () => true,
          () => false
        );
      const committed = mainData.gameCount - initialCount;
      const targetReached = committed >= parseInt(targetGames, 10);

      // If we've reached the target and there is a buffered tail, flush it now
      if (targetReached && toCommit.length > 0) {
        await maybeCommit(true);
      }

      // Commit if writers finished or if >1s since last commit, or buffer >= 2
      if (
        toCommit.length >= 2 ||
        writersFinished ||
        (Date.now() - lastCommitAt > 1000 && toCommit.length > 0)
      ) {
        await maybeCommit(true);
      }

      // Exit early if target reached and buffer empty
      if (targetReached && toCommit.length === 0) {
        clearInterval(pollInterval);
        await maybeCommit(true);
        console.log(`Consolidator complete: ${mainData.gameCount} total games`);
        process.exit(0);
      }

      // Exit criteria with sentinel, EOF check, and empty buffer
      const allFilesFullyRead = await checkAllOffsetsAtEOF(workers, runDir);
      if (writersFinished && allFilesFullyRead && toCommit.length === 0) {
        clearInterval(pollInterval);
        await maybeCommit(true); // final commit
        console.log(`Consolidator complete: ${mainData.gameCount} total games`);
        process.exit(0);
      }
    } catch (err) {
      console.error('Error in polling loop:', err);
    }
  }, 500);

  // Graceful shutdown
  process.on('SIGINT', async () => {
    console.log('Consolidator shutting down...');
    clearInterval(pollInterval);
    await maybeCommit(true);
    process.exit(0);
  });
}

main().catch((err) => {
  console.error('Error in consolidator:', err);
  process.exit(1);
});
