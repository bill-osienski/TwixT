#!/usr/bin/env node
/**
 * Helper script to run a 60/60 validation batch, capture the raw log,
 * aggregate heuristic counters, compute win/draw splits, and append a
 * summary entry to logs/validation-results.json.
 */

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';
import crypto from 'crypto';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');

function stableStringify(value) {
  if (Array.isArray(value)) {
    return '[' + value.map((v) => stableStringify(v)).join(',') + ']';
  }
  if (value && typeof value === 'object') {
    return (
      '{' +
      Object.keys(value)
        .sort()
        .map((key) => `${JSON.stringify(key)}:${stableStringify(value[key])}`)
        .join(',') +
      '}'
    );
  }
  return JSON.stringify(value);
}

const DEFAULT_DEPTH_CONFIG = '2:60,3:60';
const DEFAULT_WORKERS = 10;
const OPTIONAL_OFFENSE_KEYS = [
  'connectorBonus',
  'finishThreshold',
  'finishBonusBase',
  'connectorTargetBonus',
  'doubleCoverageBase',
  'finishGapSlope',
  'nearFinishBonus',
  'redFinishExtra',
  'redGapDecayMultiplier',
];

function parseArgs(argv) {
  const options = {
    log: null,
    depthConfig: DEFAULT_DEPTH_CONFIG,
    workers: DEFAULT_WORKERS,
  };

  for (const arg of argv) {
    if (!arg.startsWith('--')) continue;
    const [key, rawValue] = arg.slice(2).split('=', 2);
    const value = rawValue ?? 'true';
    switch (key) {
      case 'log':
        options.log = value;
        break;
      case 'depth-config':
        options.depthConfig = value;
        break;
      case 'workers':
        options.workers = Number(value);
        break;
      default:
        console.warn(`Unknown argument --${key}, ignoring.`);
    }
  }

  return options;
}

function ensureDir(dirPath) {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
}

function readJSON(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function writeJSON(filePath, data) {
  fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
}

function aggregateHeuristics(logText) {
  const regex = /\[TwixTAI\] heuristic stats (\{[\s\S]*?\})(?=\n)/g;
  const entries = [];
  let match;
  while ((match = regex.exec(logText)) !== null) {
    try {
      entries.push(JSON.parse(match[1]));
    } catch (err) {
      console.warn('Failed to parse heuristic entry:', err.message);
    }
  }

  const totals = {};
  for (const entry of entries) {
    for (const depth of Object.keys(entry)) {
      const heur = entry[depth];
      for (const [name, val] of Object.entries(heur)) {
        if (!totals[name]) {
          totals[name] = {
            red: { count: 0, sum: 0 },
            black: { count: 0, sum: 0 },
          };
        }
        if (val.red) {
          totals[name].red.count += val.red.count || 0;
          totals[name].red.sum += val.red.sum || 0;
        }
        if (val.black) {
          totals[name].black.count += val.black.count || 0;
          totals[name].black.sum += val.black.sum || 0;
        }
      }
    }
  }

  return { entryCount: entries.length, totals };
}

function tallyGames(games) {
  const result = { red: 0, black: 0, draw: 0 };
  for (const game of games) {
    const summary = game.summary || {};
    if (summary.draw) {
      result.draw += 1;
    } else if (summary.winner === 'red') {
      result.red += 1;
    } else if (summary.winner === 'black') {
      result.black += 1;
    }
  }
  return result;
}

function sliceGamesSegment(allGames, startIndex) {
  if (startIndex < 0 || startIndex > allGames.length) {
    throw new Error('Invalid startIndex when slicing games segment.');
  }
  return allGames.slice(startIndex);
}

function snapshotConfig() {
  const searchPath = path.join(projectRoot, 'assets/js/ai/search.json');
  const config = readJSON(searchPath);
  const offense = config.rewards?.edge?.offense || {};
  const snapshot = {
    firstEdgeTouchRed: offense.firstEdgeTouchRed,
    firstEdgeTouchBlack: offense.firstEdgeTouchBlack,
    finishPenaltyBase: offense.finishPenaltyBase,
    redFinishPenaltyFactor: offense.redFinishPenaltyFactor,
    blackFinishScaleMultiplier: offense.blackFinishScaleMultiplier,
    redSpanGainMultiplier: offense.redSpanGainMultiplier,
    blackSpanGainMultiplier: offense.blackSpanGainMultiplier,
    redDoubleCoverageBonus: offense.redDoubleCoverageBonus,
    blackDoubleCoverageScale: offense.blackDoubleCoverageScale,
  };
  for (const key of OPTIONAL_OFFENSE_KEYS) {
    snapshot[key] = offense[key];
  }
  return snapshot;
}

function computeConfigHash(config) {
  const payload = {
    firstEdgeTouchRed: config.firstEdgeTouchRed,
    firstEdgeTouchBlack: config.firstEdgeTouchBlack,
    finishPenaltyBase: config.finishPenaltyBase,
    redFinishPenaltyFactor: config.redFinishPenaltyFactor,
    blackFinishScaleMultiplier: config.blackFinishScaleMultiplier,
    redSpanGainMultiplier: config.redSpanGainMultiplier,
    blackSpanGainMultiplier: config.blackSpanGainMultiplier,
    redDoubleCoverageBonus: config.redDoubleCoverageBonus,
    blackDoubleCoverageScale: config.blackDoubleCoverageScale,
  };
  for (const key of OPTIONAL_OFFENSE_KEYS) {
    payload[key] = config[key];
  }
  const json = stableStringify(payload);
  return crypto.createHash('sha1').update(json).digest('hex');
}

function runValidation(options) {
  const logsDir = path.join(projectRoot, 'logs');
  ensureDir(logsDir);

  const timestamp = new Date().toISOString();
  const label = options.label ?? `validation-${timestamp}`;

  const logFileName =
    options.log && !options.log.includes('/') && !options.log.includes('\\')
      ? options.log
      : `validation-${timestamp.replace(/[:.]/g, '-')}.log`;
  const logFilePath = path.isAbsolute(logFileName)
    ? logFileName
    : path.join(logsDir, logFileName);

  const configSnapshot = snapshotConfig();
  const configHash = computeConfigHash(configSnapshot);

  const selfPlayPath = path.join(projectRoot, 'selfplay.json');
  const beforeData = readJSON(selfPlayPath);
  const beforeLength = beforeData.games?.length || 0;

  const command = [
    'node',
    'scripts/selfPlayParallel.js',
    '--depth-config',
    `"${options.depthConfig}"`,
    '--workers',
    String(options.workers),
    '--verbose',
  ].join(' ');

  console.log(`Running validation batch: ${command}`);
  const output = execSync(command, {
    cwd: projectRoot,
    stdio: 'pipe',
    maxBuffer: 1024 * 1024 * 10, // 10 MB
  }).toString('utf8');

  fs.writeFileSync(logFilePath, output);
  console.log(
    `Validation log written to ${path.relative(projectRoot, logFilePath)}`
  );

  const runIdMatch = output.match(/Run ID:\s*(\d+)/);
  const runId = runIdMatch ? runIdMatch[1] : null;

  const afterData = readJSON(selfPlayPath);
  const games = afterData.games || [];
  const segment = sliceGamesSegment(games, beforeLength);

  if (segment.length === 0) {
    throw new Error('No new games were recorded. Aborting.');
  }

  // Assume first half depth-2, second half depth-3.
  const half = Math.floor(segment.length / 2);
  const depth2Games = segment.slice(0, half);
  const depth3Games = segment.slice(half);

  const statsOverall = tallyGames(segment);
  const statsDepth2 = tallyGames(depth2Games);
  const statsDepth3 = tallyGames(depth3Games);

  const heuristics = aggregateHeuristics(output);

  const validationSummary = {
    timestamp,
    runId,
    configHash,
    command: {
      depthConfig: options.depthConfig,
      workers: options.workers,
    },
    config: configSnapshot,
    gamesRecorded: segment.length,
    wins: statsOverall,
    depth2: statsDepth2,
    depth3: statsDepth3,
    heuristics,
  };

  const validationSummaryPath = path.join(logsDir, 'validation-results.json');
  const existing = fs.existsSync(validationSummaryPath)
    ? readJSON(validationSummaryPath)
    : { runs: [] };
  existing.runs.push(validationSummary);
  writeJSON(validationSummaryPath, existing);

  console.log('Validation summary appended to logs/validation-results.json');
}

const options = parseArgs(process.argv.slice(2));

try {
  runValidation(options);
} catch (err) {
  console.error('Validation run failed:', err.message);
  process.exit(1);
}
