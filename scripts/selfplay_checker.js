#!/usr/bin/env node
/**
 * Self-play file validator + depth stats (draw-aware)
 * - Validates structure/fields (top-level + per-game + per-move)
 * - Detects duplicates, non-sequential gameNumbers, legacy heavy fields, OOB moves
 * - Prints depth buckets (d1/d2/d3/unknown) with win/move/draw stats
 *
 * Usage:
 *   node scripts/selfplay_checker.js
 *   node scripts/selfplay_checker.js /path/to/other-selfplay.json
 * Options:
 *   --strict         Exit with non-zero code if issues_total > 0
 *   --max-samples N  Limit number of example findings shown (default 0 = none)
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// CLI parsing (simple)
const args = process.argv.slice(2);
const flags = new Set(args.filter((a) => a.startsWith('--')));
const argPath = args.find((a) => !a.startsWith('-'));
const STRICT = flags.has('--strict');

let maxSamples = 0;
const maxIdx = args.findIndex((a) => a === '--max-samples');
if (maxIdx !== -1 && args[maxIdx + 1]) {
  maxSamples = Math.max(0, parseInt(args[maxIdx + 1], 10) || 0);
}

// Default input is one level up (project root)
const defaultPath = path.resolve(__dirname, '..', 'selfplay.json');
const inputPath = argPath ? path.resolve(process.cwd(), argPath) : defaultPath;

function loadJson(p) {
  try {
    const raw = fs.readFileSync(p, 'utf8');
    return JSON.parse(raw);
  } catch (err) {
    console.error(`Failed to read/parse "${p}": ${err.message}`);
    process.exit(1);
  }
}

const isInt = (x) => Number.isInteger(x);
const isFiniteNum = (x) => Number.isFinite(x);
const isPlayer = (s) => s === 'red' || s === 'black';

function main() {
  const data = loadJson(inputPath);

  const issues = {
    duplicates: 0,
    nonSequential: 0,
    badSummary: 0,
    badMoves: 0,
    badMeta: 0,
    legacyBoards: 0,
    legacyTraces: 0,
    movesVsSummaryMismatch: 0,
    outOfBoundsMoves: 0,
    emptyGames: 0,
    badDrawConsistency: 0,
  };
  const sampleFindings = [];

  const games = Array.isArray(data.games) ? data.games : [];
  const top_gameCount = isInt(data.gameCount) ? data.gameCount : null;

  // Duplicate detection
  const seenGameNumbers = new Set();
  const seenProvenance = new Set(); // runId-coreId-seq

  // Depth buckets (with fallback to root searchDepth)
  const rootDepth = Number.isFinite(data.searchDepth) ? data.searchDepth : null;
  const buckets = new Map(); // depthKey => { games, redWins, blackWins, draws, totalMoves }
  const keyFor = (g) => {
    const d = g?.meta?.depth;
    if (Number.isFinite(d)) return `d${d}`;
    if (Number.isFinite(rootDepth)) return `d${rootDepth}`;
    return 'unknown';
  };
  const bumpBucket = (k, g) => {
    if (!buckets.has(k)) {
      buckets.set(k, {
        games: 0,
        redWins: 0,
        blackWins: 0,
        draws: 0,
        totalMoves: 0,
      });
    }
    const b = buckets.get(k);
    b.games++;
    const s = g?.summary;
    if (s?.winner === 'red') b.redWins++;
    else if (s?.winner === 'black') b.blackWins++;
    else if (s && (s.draw === true || s.winner == null)) b.draws++;
    b.totalMoves += s?.totalMoves || g.moves?.length || 0;
  };

  // Sequential checks
  let expectGameNumber = 1;
  for (let idx = 0; idx < games.length; idx++) {
    const g = games[idx];

    // --- Summary checks ---
    const s = g?.summary;
    let badSummary = false;
    if (!s || typeof s !== 'object') badSummary = true;
    if (!isInt(s?.boardSize) || s.boardSize <= 0) badSummary = true;
    if (!isInt(s?.totalMoves) || s.totalMoves < 0) badSummary = true;
    if (typeof s?.gameOver !== 'boolean') badSummary = true;
    // Winner/Draw consistency
    const hasWinner = isPlayer(s?.winner);
    const hasDraw = s?.draw === true;
    if (hasWinner && hasDraw) {
      issues.badDrawConsistency++;
      if (sampleFindings.length < maxSamples) {
        sampleFindings.push({
          at: `gameIndex=${idx}`,
          issue: 'winner_and_draw_both_set',
          summary: s,
        });
      }
    }
    if (!hasWinner && !hasDraw && s?.gameOver === true) {
      // game over but neither winner nor draw flagged — allow (legacy), but mark as summary issue
      issues.badDrawConsistency++;
      if (sampleFindings.length < maxSamples) {
        sampleFindings.push({
          at: `gameIndex=${idx}`,
          issue: 'gameOver_without_winner_or_draw',
          summary: s,
        });
      }
    }
    if (!hasWinner && s?.winner != null && !hasDraw) {
      badSummary = true; // winner is non-null but not 'red'/'black'
    }
    if (badSummary) {
      issues.badSummary++;
      if (sampleFindings.length < maxSamples) {
        sampleFindings.push({
          at: `gameIndex=${idx}`,
          issue: 'badSummary',
          summary: s,
        });
      }
    }

    // --- Meta checks ---
    const m = g?.meta;
    let badMeta = false;
    if (m != null && typeof m === 'object') {
      if (typeof m.runId !== 'string' || m.runId.length === 0) badMeta = true;
      if (!isFiniteNum(m.coreId)) badMeta = true;
      if (!isFiniteNum(m.seq)) badMeta = true;
      if (!isFiniteNum(m.depth)) badMeta = true;
      if (typeof m.createdAt !== 'string' || m.createdAt.length === 0)
        badMeta = true;

      const key = `${m.runId}-${m.coreId}-${m.seq}`;
      if (seenProvenance.has(key)) {
        issues.duplicates++;
        if (sampleFindings.length < maxSamples) {
          sampleFindings.push({
            at: `gameIndex=${idx}`,
            issue: 'duplicate_provenance',
            key,
          });
        }
      } else {
        seenProvenance.add(key);
      }
    } else {
      // Older games may not have meta; still count as badMeta
      badMeta = true;
    }
    if (badMeta) {
      issues.badMeta++;
      if (sampleFindings.length < maxSamples) {
        sampleFindings.push({
          at: `gameIndex=${idx}`,
          issue: 'badMeta',
          meta: m,
        });
      }
    }

    // --- gameNumber checks (sequential & duplicates) ---
    const gn = g?.gameNumber;
    if (!isInt(gn) || gn < 1) {
      issues.nonSequential++;
      if (sampleFindings.length < maxSamples) {
        sampleFindings.push({
          at: `gameIndex=${idx}`,
          issue: 'missing_or_bad_gameNumber',
          gameNumber: gn,
        });
      }
    } else {
      if (seenGameNumbers.has(gn)) {
        issues.duplicates++;
        if (sampleFindings.length < maxSamples) {
          sampleFindings.push({
            at: `gameIndex=${idx}`,
            issue: 'duplicate_gameNumber',
            gameNumber: gn,
          });
        }
      }
      seenGameNumbers.add(gn);
      if (gn !== expectGameNumber) {
        issues.nonSequential++;
        if (sampleFindings.length < maxSamples) {
          sampleFindings.push({
            at: `gameIndex=${idx}`,
            issue: 'nonSequential',
            expected: expectGameNumber,
            found: gn,
          });
        }
        expectGameNumber = gn + 1;
      } else {
        expectGameNumber++;
      }
    }

    // --- Moves checks ---
    const moves = Array.isArray(g?.moves) ? g.moves : null;
    if (!moves || moves.length === 0) {
      issues.emptyGames++;
      if (sampleFindings.length < maxSamples) {
        sampleFindings.push({ at: `gameIndex=${idx}`, issue: 'no_moves' });
      }
    } else {
      // Legacy heavy fields?
      for (let j = 0; j < moves.length; j++) {
        const mv = moves[j];
        if (Object.prototype.hasOwnProperty.call(mv, 'board')) {
          issues.legacyBoards++;
          if (sampleFindings.length < maxSamples) {
            sampleFindings.push({
              at: `gameIndex=${idx}, move=${j}`,
              issue: 'legacy_board_present',
            });
          }
          break;
        }
      }
      for (let j = 0; j < moves.length; j++) {
        const mv = moves[j];
        if (Object.prototype.hasOwnProperty.call(mv, 'lastMoveTrace')) {
          issues.legacyTraces++;
          if (sampleFindings.length < maxSamples) {
            sampleFindings.push({
              at: `gameIndex=${idx}, move=${j}`,
              issue: 'legacy_lastMoveTrace_present',
            });
          }
          break;
        }
      }

      // Detailed move validation (lightweight)
      const boardSize = s?.boardSize || 24;
      for (let j = 0; j < moves.length; j++) {
        const mv = moves[j];
        let bad = false;

        if (!isInt(mv?.turn) || mv.turn < 0) bad = true;
        if (!isPlayer(mv?.player)) bad = true;

        const pos = mv?.move;
        if (!pos || !isInt(pos.row) || !isInt(pos.col)) bad = true;
        if (
          pos &&
          (pos.row < 0 ||
            pos.row >= boardSize ||
            pos.col < 0 ||
            pos.col >= boardSize)
        ) {
          issues.outOfBoundsMoves++;
          if (sampleFindings.length < maxSamples) {
            sampleFindings.push({
              at: `gameIndex=${idx}, move=${j}`,
              issue: 'out_of_bounds',
              move: pos,
              boardSize,
            });
          }
        }

        if (mv?.heuristicScore != null && !isFiniteNum(mv.heuristicScore))
          bad = true;

        if (bad) issues.badMoves++;
      }

      // Moves vs summary count
      const declared = s?.totalMoves;
      const actual = moves.length;
      if (isInt(declared) && declared !== actual) {
        issues.movesVsSummaryMismatch++;
        if (sampleFindings.length < maxSamples) {
          sampleFindings.push({
            at: `gameIndex=${idx}`,
            issue: 'moves_vs_summary_mismatch',
            declared,
            actual,
          });
        }
      }
    }

    // Depth bucket bump
    bumpBucket(keyFor(g), g);
  }

  // Build report
  const issues_total = Object.values(issues).reduce((a, b) => a + (b || 0), 0);

  // Depth table
  let totalGames = 0;
  for (const b of buckets.values()) totalGames += b.games;
  const depthRows = Array.from(buckets.entries())
    .sort((a, b) => {
      const aNum = a[0].startsWith('d')
        ? parseInt(a[0].slice(1), 10)
        : Number.POSITIVE_INFINITY;
      const bNum = b[0].startsWith('d')
        ? parseInt(b[0].slice(1), 10)
        : Number.POSITIVE_INFINITY;
      return aNum - bNum;
    })
    .map(([depth, b]) => {
      const decided = Math.max(1, b.redWins + b.blackWins); // avoid /0 in rate when printing
      const nonDrawGames = b.games - b.draws || 0;
      const redRate = nonDrawGames
        ? +((100 * b.redWins) / nonDrawGames).toFixed(1)
        : 0;
      const blackRate = nonDrawGames
        ? +((100 * b.blackWins) / nonDrawGames).toFixed(1)
        : 0;
      const drawRate = b.games ? +((100 * b.draws) / b.games).toFixed(1) : 0;

      return {
        depth,
        games: b.games,
        pct: +((100 * b.games) / (totalGames || 1)).toFixed(1),
        avgMoves: b.games ? +(b.totalMoves / b.games).toFixed(1) : 0,
        redWins: b.redWins,
        blackWins: b.blackWins,
        draws: b.draws,
        redWinRate: redRate,
        blackWinRate: blackRate,
        drawRate,
      };
    });

  // Summary JSON
  const report = {
    file: inputPath,
    games: games.length,
    top_gameCount,
    gameCount_matches_top:
      top_gameCount == null ? null : top_gameCount === games.length,
    issues_total,
    ...issues,
    sampleFindings, // limited by --max-samples
    depthBuckets: depthRows, // for programmatic use
  };

  // Pretty output
  console.log(`Reading: ${inputPath}\n`);
  console.log('Depth breakdown:');
  console.table(depthRows);
  console.log('\nValidation summary:');
  console.log(JSON.stringify(report, null, 2));

  if (STRICT && issues_total > 0) {
    process.exitCode = 2;
  }
}

main();
